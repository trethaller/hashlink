# JIT Store Optimization -- Problem, Proposed Fix, and Risks

## The Problem

### Observed Symptoms (Intel VTune)

Profiling the HashLink JIT on a hot arithmetic loop revealed:

- **Back-End Bound: 41.3%** of Pipeline Slots
- **Memory Bound: 36.8%** of Pipeline Slots
- **Store Bound: 34.9%** of Clockticks -- dominant bottleneck
- **L1/L2/L3/DRAM Bound: ~0.3%** each -- data is NOT missing cache

The store pressure is not caused by cache misses. It is caused by the sheer volume of store
instructions filling the CPU's store buffer (a finite hardware queue, ~20-56 entries). When the
store buffer saturates, the back-end allocator stalls and cannot dispatch new micro-ops.

### Root Cause: Write-Through Stack Policy in `store()`

The HashLink JIT uses a **write-through** register allocation policy. Every vreg has:
- A fixed stack slot (`vreg.stack`, kind `RSTACK`, address `[rbp - N]`) -- the authoritative home
- An optional physical register (`vreg.current`) -- a transient cache

After every bytecode op that writes a vreg, `store()` is called (56 call sites). `store()`
unconditionally emits a `movsd [rbp-N], xmmX` instruction, keeping the stack slot current:

```c
static void store( jit_ctx *ctx, vreg *r, preg *v, bool bind ) {
    ...
    v = copy(ctx, &r->stack, v, r->size);  // always writes to stack slot
    ...
}
```

For a simple expression like `result = (float(x) / CONST) * a + b`, this generates:

```asm
cvtsi2sd xmm5, edx
movsd [rbp-0x20], xmm5    ; dead -- overwritten before any read
movsd xmm0, [rip+...]
movsd [rbp-0x50], xmm0    ; dead -- overwritten before any read
divsd xmm5, xmm0
movsd [rbp-0x20], xmm5    ; dead -- overwritten before any read
movsd xmm2, [r10+0x8]
movsd [rbp-0x50], xmm2    ; dead -- overwritten before any read
mulsd xmm5, xmm2
movsd [rbp-0x20], xmm5    ; dead -- overwritten before any read
```

Every store to `[rbp-0x20]` and `[rbp-0x50]` is a dead write. The computation is entirely
register-to-register. The stack writes exist purely to satisfy the JIT's bookkeeping invariant.

---

## Proposed Fix: Lazy Write-Back (Dirty Flag)

Convert the register policy from **write-through** to **write-back**: defer the actual stack
store until the value genuinely needs to be in memory.

### Core invariant

> Always flush a dirty vreg before dropping its register binding.

If this holds, `fetch()` is already correct as-is: when `r->current` is non-NULL it returns the
register (dirty or not, doesn't matter). When `r->current` is NULL, the binding was already
dropped, so the stack is guaranteed current. No changes to `fetch()` are needed.

### Why `scratch()` is the critical point

There are 3 code patterns that drop register bindings:

1. **`scratch()`** -- 45 call sites throughout the file
2. **`alloc_reg()` eviction** -- 2 sites that inline the drop instead of calling `scratch()`
3. **`discard_regs()`** -- 1 function (8 call sites) that loops over all registers

`scratch()` was designed as a **pure metadata operation** -- it clears pointers but never emits
instructions. Almost every binding drop in the entire JIT flows through it. Making it flush-aware
covers the vast majority of code paths automatically.

We cannot just add flush calls at the "structural choke points" (`alloc_reg`
eviction, `save_regs`) and ignore `scratch()`. Example: line 1742 scratches `Ecx` to free it for
a shift operation. Whatever dirty vreg was in `Ecx` loses its value permanently. A later `fetch()`
on that vreg returns the stale stack slot. This is not a theoretical concern -- it happens in any
code mixing arithmetic with shifts/divisions.

### The macro approach

Every function in `jit.c` that calls `scratch()` already has `jit_ctx *ctx` in scope. Rename the
function, add flush, redefine the old name as a macro:

```c
static void scratch_impl( jit_ctx *ctx, preg *r ) {
    if( r && r->holds ) {
        flush_vreg(ctx, r->holds);
        r->holds->current = NULL;
        r->holds = NULL;
        r->lock = 0;
    }
}
#define scratch(r) scratch_impl(ctx, r)
```

This changes zero call sites. The macro captures `ctx` from the enclosing scope.

Alternative: add `ctx` as an explicit parameter and mechanically update all 45 call sites
(`scratch(x)` -> `scratch(ctx, x)`). More principled, larger diff, same result.

### Known issue: `OAsm` case 3 clobber

There is exactly one `scratch()` call site where the caller has already written to the stack
before calling `scratch()`:

```c
case 3: // write vm reg
    rb--;
    copy(ctx, &rb->stack, REG_AT(o->p2), rb->size);  // writes NEW value to stack
    scratch(rb->current);  // flush writes OLD value, clobbering the new one
```

With flush-on-scratch, step 2 overwrites the new value with the old one. Fix: swap the two lines
so scratch (and its flush) happens first:

```c
case 3: // write vm reg
    rb--;
    scratch(rb->current);  // flush old value if dirty, then drop binding
    copy(ctx, &rb->stack, REG_AT(o->p2), rb->size);  // write new value
```

This is the ONLY call site with this pattern. All other scratch() sites are safe:
- ~38 sites: scratch evicts a vreg whose value matters -- flush is correct and necessary
- 2 sites (lines 1097, 1148): value was just MOV'd to another register -- flush is redundant
  but harmless (writes same value to stack, then vreg is rebound to the new register)
- 3 sites (lines 1295, 1299, 1303): `store_result` x86-32 -- flush writes old value, then FSTP
  immediately overwrites with new value. Redundant but harmless.

### `reg_bind()` does NOT need changes

`reg_bind()` drops the old register's `holds` pointer at line 1062 without going through
`scratch()`. However, it immediately sets `r->current = p` -- the vreg never becomes unbound.
It transitions from one register to another. The dirty flag carries over correctly: if the vreg
was dirty in the old register, it's still dirty in the new one (both hold the same value since
the caller already MOV'd it). No flush needed.

---

## All Changes

### 1. Add `dirty` to `vreg`

```c
struct vreg {
    int stackPos;
    int size;
    hl_type *t;
    preg *current;
    preg stack;
    bool dirty;
};
```

Initialize to `false` in the vreg setup loop (~line 2941).

### 2. Add `flush_vreg` helper

```c
static void flush_vreg( jit_ctx *ctx, vreg *r ) {
    if( r->dirty && r->current ) {
        copy(ctx, &r->stack, r->current, r->size);
        r->dirty = false;
    }
}
```

Note: `flush_vreg` calls `copy()` with `to=RSTACK`, `from=RCPU/RFPU`. These copy paths emit a
direct MOV/MOVSD instruction and never call `alloc_reg()`, so there is no re-entrancy risk.

### 3. Make `scratch()` flush-aware (macro rename)

```c
static void scratch_impl( jit_ctx *ctx, preg *r ) {
    if( r && r->holds ) {
        flush_vreg(ctx, r->holds);
        r->holds->current = NULL;
        r->holds = NULL;
        r->lock = 0;
    }
}
#define scratch(r) scratch_impl(ctx, r)
```

### 4. Keep `discard_regs()` pure

`discard_regs()` must remain a pure metadata drop.

It is called:
- after native and HL calls
- at jump targets / merge points during code generation

If `discard_regs()` emits flush stores there, those stores would run after caller-saved registers
may already be clobbered by the call, or at a merge point where the compile-time binding state does
not necessarily match the runtime predecessor path. That can write garbage or wrong-path values to
the stack.

So `discard_regs()` should stay exactly as it is today: clear bindings only, emit no code.

### 5. Add flush in `alloc_reg()` eviction

The eviction loops inline the binding drop instead of calling `scratch()`. Two sites (CPU line
~973, FPU line ~997). Add `flush_vreg(ctx, p->holds)` before each drop. Or replace the inline
drop with `scratch(p)` to get the flush automatically (scratch sets `lock=0`, but the subsequent
`RLOCK(p)` immediately overrides it -- safe).

### 6. Add `flush_all_dirty()` helper and call before calls, jumps, and merge points

Because `discard_regs()` stays pure, every site that will lose register state must flush before
that happens. There are three categories: calls, branches, and fallthrough into merge points.

```c
static void flush_all_dirty( jit_ctx *ctx ) {
    int i;
    for(i=0;i<RCPU_SCRATCH_COUNT;i++) {
        preg *r = ctx->pregs + RCPU_SCRATCH_REGS[i];
        if( r->holds ) flush_vreg(ctx, r->holds);
    }
    for(i=0;i<RFPU_COUNT;i++) {
        preg *r = ctx->pregs + XMM(i);
        if( r->holds ) flush_vreg(ctx, r->holds);
    }
}
```

**6a. Before calls:**

Call it immediately before:
- `op_call(...)` in `call_native()`
- `op_call(...)` in the non-native path of `op_call_fun()`
- any other direct call-emission site that is followed by `discard_regs()`

**6b. Before branch instructions:**

All branches go through `do_jump()` (a central function) or a handful of direct `XJump` macro
uses that feed into `register_jump()`. At the jump target, `discard_regs()` will drop all
register bindings. If any vreg was dirty at the source, the target's `fetch()` returns a stale
stack slot.

Add `flush_all_dirty(ctx)` at the top of `do_jump()`. For the few `XJump` calls that feed
directly into `register_jump()` without going through `do_jump()` (e.g., OJFalse/OJTrue/OJNull
at line ~3135, OSwitch at line ~4444), add `flush_all_dirty(ctx)` before those `XJump` calls.

There are 23 `register_jump()` call sites total. Most go through `do_jump()` which centralizes
the flush. The remaining handful of direct `XJump` → `register_jump` patterns need individual
`flush_all_dirty()` calls.

**6c. Before fallthrough into merge points:**

When linear code falls through into a position that is a jump target, the code at line ~4552
calls `discard_regs()`. A fallthrough flush is needed so that the stack is current before
bindings are dropped. Add `flush_all_dirty(ctx)` before `discard_regs` at line ~4552 and before
`discard_regs` at `OLabel` (line ~3844).

These fallthrough flushes are emitted into the code stream BEFORE `BUF_POS()` is recorded (line
~4553), so jump arrivals skip them — they land at `BUF_POS()`, which is after the flush code.
Only fallthrough executes the flush. This is correct: jump sources flush at their own site (6b),
fallthrough flushes here (6c).

### 7. Fix `OAsm` case 3 ordering

Swap the two lines so scratch/flush happens before the direct stack write (see above).

### 8. Guard direct `&r->stack` reads

A handful of places bypass `fetch()` and use `&r->stack` directly as a memory operand. If
`r->current` is non-NULL and dirty, the stack slot is stale. Add `flush_vreg(ctx, r)` before each:

| Site | Context | Platform |
|---|---|---|
| `ORef` (line ~3999) | `LEA [rbp+stackPos]` passed to native code | All |
| `make_dyn_cast` (line ~2896) | Same pattern | All |
| `OCallMethod` HVIRTUAL (line ~3761) | `LEA &a->stack` for `hl_dyn_call_obj` | All |
| `OToDyn` (line ~3177) | `MOV tmp, &ra->stack` | x86-32 only |
| `OAsm` case 2 (line ~4522) | `copy(..., &rb->stack, ...)` | All |
| `op_ret` (lines ~1625, ~1632) | `FLD &r->stack` | x86-32 only |
| `push_reg` (lines ~1408, ~1410) | `PUSH &r->stack` | x86-32 only |

The `push_reg` and `op_ret` sites are preceded by `scratch(r->current)`, which now flushes via
the macro. So they are already covered by step 3 -- the explicit `flush_vreg` call is not needed
there, but adding one anyway is harmless and defensive.

### 9. Flush before `save_regs()`

`save_regs()` snapshots register bindings but not dirty flags. `restore_regs()` restores the
old bindings but leaves dirty flags from the alternate code path. Fix: flush all dirty vregs
before `save_regs()` so the snapshot starts clean. Also clear all dirty flags inside
`restore_regs()` (one line: `ctx->vregs[i].dirty = false` in the existing nulling loop).

There are exactly 2 `save_regs()` call sites: `OCallClosure` (line ~3454) and `OCallMethod`
HVIRTUAL (line ~3736).

### 10. Modify `store()` to defer the write

```c
static void store( jit_ctx *ctx, vreg *r, preg *v, bool bind ) {
    if( r->current && r->current != v ) {
        r->current->holds = NULL;
        r->current = NULL;
    }
    if( bind && (v->kind == RCPU || v->kind == RFPU) ) {
        if( r->current != v ) {
            scratch(v);
            r->current = v;
            v->holds = r;
        }
        r->dirty = true;
    } else {
        copy(ctx, &r->stack, v, r->size);
        r->dirty = false;
    }
}
```

**CRITICAL: the `r->current != v` check must be preserved from the original code (line 1284).**

Without it, when the same vreg is written repeatedly using the same register (the hot path —
consecutive arithmetic like `divsd xmm5, ...` / `store(r, xmm5, true)` / `mulsd xmm5, ...` /
`store(r, xmm5, true)`), `scratch(v)` fires on every call. Since `v->holds == r`, the scratch
flushes `r` to the stack — emitting the exact store we're trying to eliminate. The optimization
becomes a no-op for the most important case.

With the check: when `r->current == v` (vreg already bound to this register), we skip
scratch/rebind and just set `dirty = true`. No code emitted. This is the path that actually
saves the stores.

The original code had both conditions in one `if`:
```c
if( bind && r->current != v && (v->kind == RCPU || v->kind == RFPU) )
```

Our version separates the two concerns: the outer `if` decides register-vs-stack policy, the
inner `if` decides whether rebinding is needed. `dirty = true` is always set in the register path.

Note: the original `store()` had an assertion `IS_FLOAT(r) != (v->kind == RFPU)`. This check
should be preserved in the new code for safety during development.

---

## Risks and Concerns

### The previous attempt crashed with random corruption

This is the biggest concern. Random corruption means a flush point was missed or a flush happened
at the wrong time (clobbering a value). The macro approach for `scratch()` is specifically designed
to prevent the "missed flush" class of bugs -- every binding drop automatically flushes. The OAsm
case 3 clobber is the one known case of "flush at the wrong time" and is trivially fixed.

A likely cause of the previous crash: the `r->current != v` check was missing from `store()`.
Without it, every `store()` call on a vreg that's already bound to the target register triggers
scratch → flush → rebind on every operation. This is a behavioral change the original code never
had — the original skipped the scratch/rebind entirely when `r->current == v` (line 1284). The
repeated scratch/rebind cycle could corrupt lock states, emit stores at unexpected points in the
instruction stream, and cause the register allocator to see inconsistent binding state.

### Failure modes

| Mode | Cause | Symptom | Severity |
|---|---|---|---|
| Missed flush | Binding dropped without flushing dirty vreg | Wrong value read from stack later | Silent corruption |
| Over-flush | Flush writes old value over a direct stack write | New value lost | Silent corruption |
| Stale dirty flag | `dirty=true` but value is actually in stack | Redundant flush emitted | Performance waste only |
| False clean | `dirty=false` but stack is actually stale | No flush when needed | Silent corruption |

The macro approach eliminates "missed flush" for all 45 scratch() sites. "Over-flush" has exactly
1 known instance (OAsm case 3, fixed by reordering). "Stale dirty flag" is harmless. "False clean"
can only happen if something writes to a register without going through `store()` -- review needed
but no such path has been identified.

### `flush_vreg` inside `alloc_reg()` is safe from re-entrancy

`flush_vreg` calls `copy(ctx, &r->stack, r->current, r->size)`. The destination is always RSTACK
and the source is always RCPU or RFPU. The `copy()` path for these combinations emits a direct
MOV/MOVSD instruction and never calls `alloc_reg()`. No circular dependency.

### Calls, jumps, and merge points

`discard_regs()` is called after calls and at jump targets. This is exactly why it must not flush:

- After a call, caller-saved registers may already be clobbered, so a post-call flush can write
  garbage.
- At a jump target, compile-time register-binding state may not match the runtime predecessor path,
  so a target-side flush can write wrong-path values.

Flushes must happen before the operation that loses the bindings:
- before `CALL` instructions (pre-call flush)
- before branch instructions (pre-jump flush via `do_jump()`)
- before fallthrough into merge points (pre-fallthrough flush at line ~4552 and `OLabel`)

The pre-jump flush is critical and was missing from earlier drafts of this plan. Without it, a
dirty vreg at the branch source arrives at the target with a stale stack slot. This would cause
silent corruption for any code that branches with live dirty vregs — which includes every loop
back-edge after arithmetic.

### `store()` top: dropping old `r->current` without flush

The top of `store()` does:
```c
if( r->current && r->current != v ) {
    r->current->holds = NULL;
    r->current = NULL;
}
```

This drops `r`'s old register binding without flushing. This is correct because `r` is about to
receive a **new** value -- the old value in the register is no longer relevant regardless of
whether it was dirty.

---

## Summary

| What | Where | Call-site changes |
|---|---|---|
| Add `dirty` field | `vreg` struct + init loop | 0 |
| Add `flush_vreg()` | New helper | 0 |
| Rename `scratch` + macro | `scratch()` definition | 0 |
| Keep `discard_regs()` pure | No code emission inside function | 0 |
| Add flush in `alloc_reg()` eviction | 2 sites in `alloc_reg()` | 0 |
| Add `flush_all_dirty()` before calls | `call_native()` / `op_call_fun()` / similar | ~2-4 one-liners |
| Add `flush_all_dirty()` before jumps | `do_jump()` + ~3-5 direct `XJump` sites | ~4-6 one-liners |
| Add `flush_all_dirty()` at merge fallthrough | line ~4552 + `OLabel` | 2 one-liners |
| Fix `OAsm` case 3 ordering | 1 line swap | 1 |
| Guard direct `&r->stack` reads | ~5 sites | ~5 one-liners |
| Flush before `save_regs()` | 2 call sites | 2 one-liners |
| Clear dirty in `restore_regs()` | Inside function | 1 line |
| Modify `store()` to defer | `store()` body | 0 |
| **Total** | | **~17-21 one-line additions + 3 function rewrites** |
