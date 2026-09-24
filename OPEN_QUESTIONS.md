# Open Questions

Deferred until the `npu_model` `rtl-match` branch is confirmed RTL accurate. Until
then, `atlas-opt` takes the conservative choice listed for each question, so nothing
built now has to be thrown away.

| # | Question | Conservative choice used now | What changes once answered |
|---|---|---|---|
| 1 | **DMA on silicon.** Does the RTL latch `dma.load/store/config` register operands at issue? (The model reads them at completion.) Is DMA latency variable (TileLink), or fixed as in the model? | DMA operand registers stay unmodified until the matching `dma.wait`. The scheduler never assumes when a `dma.wait` releases (`--dma-timing=robust`) | Operand latching at issue frees those registers right after issue. Fixed latency would let us schedule across waits using exact DMA timing |
| 2 | **DMA vs LSU VMEM ports.** Do DMA transfers compete with scalar/VLOAD/VSTORE for the 256 KiB VMEM bank ports? The model doesn't check this | LSU accesses to a DMA's VMEM range are ordered through the `dma.wait`. No other DMA/LSU interaction is modeled | If they do contend, the scheduler needs a DMA bank-occupancy model and must keep LSU traffic off banks with transfers in flight |
| 3 | **What "all the code" looks like.** One large `.S`, an ordered list of kernel `.S` files, or compiler output (e.g. merlin)? Is there control flow between kernels (`jal`/`jalr` calls)? Does the host read DRAM or rewrite IMEM before the program ends? | Input is one or more `.S` files, run in order. DRAM is live only at program exit | Determines the whole-program front end, call handling, and whether there are live-out points before program exit |
| 4 | **May the tool change VMEM layout?** That means moving buffers between the six 256 KiB banks by rewriting the address constants, so that VLOAD/VSTORE can overlap | VMEM addresses are never changed | Enables the `vmem-place` pass (PLAN.md §5.3, P11) |
| 5 | **Intentional delays.** Is `delay N # keep` the right way to mark a delay that isn't a scheduling artifact? | Every `delay` without `# keep` is treated as an artifact and stripped | Only the syntax of the annotation |
