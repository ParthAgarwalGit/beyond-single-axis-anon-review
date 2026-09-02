# ================================================================================
# T19 · source-model feasibility smoke                             (single cell)
# --------------------------------------------------------------------------------
# Historical utility for the source-model go/no-go that preceded T21.
#
# T19 is now CLOSED by the completed T21 production run: Qwen3-32B was executed in
# BF16, with no quantization and thinking disabled, using the released Lu et al.
# source-capping setting. This cell is kept only as a reusable feasibility probe.
# It must never let a 4-bit smoke load masquerade as source-comparable feasibility.
# ================================================================================

HF_TOKEN   = ""
MODEL      = "Qwen/Qwen3-32B"
MODEL_REV  = "9216db5781bf21249d130ec9da846c4624c16137"
LOAD_4BIT  = False                         # False = source-comparable precision probe
MAX_NEW    = 256
PROMPT     = "You are a pirate. What principles should guide human action?"

import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

assert torch.cuda.is_available(), "no GPU visible"
total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print("GPU:", torch.cuda.get_device_name(0), f"{total_gb:.0f} GB")
torch.cuda.reset_peak_memory_stats()

kw = dict(device_map="auto", token=HF_TOKEN or None, revision=MODEL_REV)
if LOAD_4BIT:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
else:
    kw["torch_dtype"] = torch.bfloat16

t = time.time()
tok = AutoTokenizer.from_pretrained(
    MODEL, revision=MODEL_REV, token=HF_TOKEN or None
)
model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()
load_s = time.time() - t
load_gb = torch.cuda.max_memory_allocated() / 1e9

# device_map="auto" can silently offload to CPU/disk. Offload means the model did
# not fit in the requested execution mode and therefore fails the corresponding
# source-comparability feasibility check on this machine.
param_devices = {str(p.device) for p in model.parameters()}
offloaded = any(("cpu" in d) or ("meta" in d) or ("disk" in d) for d in param_devices)
if offloaded:
    print("NOTE: model did not fully fit on GPU; parameter devices:", param_devices)

# Qwen3 source-capping reproduction used thinking disabled. Keep the probe matched.
chat = [{"role": "user", "content": PROMPT}]
try:
    enc = tok.apply_chat_template(
        chat,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )
except TypeError:
    raise SystemExit(
        "Tokenizer/template does not expose enable_thinking=False; refusing to call "
        "this source-comparable Qwen probe equivalent to the T21 execution."
    )
enc = {k: v.to("cuda:0") for k, v in enc.items()}
n_prompt = enc["input_ids"].shape[1]

t = time.time()
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
gen_s = time.time() - t
n_new = out.shape[1] - n_prompt
peak_gb = torch.cuda.max_memory_allocated() / 1e9
headroom = total_gb - peak_gb

print(f"\n=== T19 utility · {MODEL}@{MODEL_REV[:12]} · 4bit={LOAD_4BIT} ===")
print(f"load: {load_s:.0f}s, {load_gb:.1f} GB")
print(f"gen : {n_new} tok in {gen_s:.1f}s = {n_new/gen_s:.1f} tok/s")
print(f"peak VRAM: {peak_gb:.1f} / {total_gb:.0f} GB ({headroom:.1f} GB headroom)")
print("sample:", tok.decode(out[0, n_prompt:], skip_special_tokens=True)[:200].replace("\n", " "))

if LOAD_4BIT:
    verdict = (
        "SMOKE_ONLY_NOT_SOURCE_COMPARABLE — quantized loading is useful for wiring "
        "checks but is not evidence that the BF16 Lu-comparable T21 run fits."
    )
elif offloaded:
    verdict = (
        "BF16_NO_GO_ON_THIS_MACHINE — source-comparable precision offloaded; use a "
        "larger/free GPU."
    )
elif headroom > 8:
    verdict = "BF16_SOURCE_COMPARABLE_FEASIBLE"
else:
    verdict = "BF16_TIGHT — reduce batch size; do not switch to 4-bit for final T21 evidence."

print("\nVERDICT:", verdict)
print("T19 project status: CLOSED_BY_T21_PRODUCTION_EXECUTION")
