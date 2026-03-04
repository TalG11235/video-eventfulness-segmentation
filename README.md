# MS-TCN Segmentation

## Quickstart

Train:
```bash
python3 -m src.model.crossval configs/50salads.yaml
```

Infer (per-video logits/probs to a folder):
```bash
python scripts/ms_tcn_cli.py infer --config configs/50salads_train_split1.yaml --checkpoint outputs/50salads_train_split1/best.pt --manifest data/50salads/test.split1.jsonl --output_dir outputs/ms_tcn_infer
```

Export DDTR pickle (matches `50_salads_unified.pkl` schema):
```bash
python scripts/generate_ddtr_pickle.py --config configs/50salads_train_split1.yaml --checkpoint outputs/50salads_train_split1/best.pt --manifest data/50salads/test.split1.jsonl --output outputs/ddtr/50_salads_unified.pkl
```

Dry run to preview schema:
```bash
python scripts/generate_ddtr_pickle.py --config configs/50salads_train_split1.yaml --checkpoint outputs/50salads_train_split1/best.pt --manifest data/50salads/test.split1.jsonl --dry_run
```
