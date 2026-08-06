# v4.2 Transformer Patch

These files are the changed v4.2 experiment files from
`/data/code/tessera/tessera-vlm/patch_vit_retrieval`. Copy them into the
same-named paths of the original `patch_vit_retrieval` module.

The patch retains the v4.2 data split, text tower, losses, and evaluation.
It replaces the visual MLP plus attention-pooling path with a four-block
spatial Transformer that emits a CLS global vector and 16x16 local tokens.

Run the full experiment after copying the files:

```bash
CUDA_VISIBLE_DEVICES=0 python -m patch_vit_retrieval.train_hierarchical_v42 \
  --config patch_vit_retrieval/config_transformer_v4_2.yaml

CUDA_VISIBLE_DEVICES=0 python -m patch_vit_retrieval.evaluate_hierarchical_v42 \
  --config patch_vit_retrieval/config_transformer_v4_2.yaml
```

The locked-test comparison is recorded in
`artifacts/audits/result/MODEL_COMPARISON.md`.
