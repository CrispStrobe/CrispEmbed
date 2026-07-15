# Issue draft

## Summary

`nomic-embed-text-v2-moe` fails to load even though the GGUF file itself is valid.

## Reproduce

Run the CLI directly against the model:

```powershell
& 'C:\projects\CrispEmbed\build-openblas-native-20260630-static\crispembed.exe' `
  -m 'C:\Users\winePad\Downloads\nomic-embed-text-v2-moe.Q4_K_M.gguf' `
  --json `
  'hello world'
```

## Expected

The command returns one embedding result.

## Actual

Model loading aborts with a missing tensor error like:

```text
missing required tensor ... attn.q.weight
```

## Minimum proof

- Before: the command above exits during model load with a missing-tensor failure.
- After: the same command returns JSON with an embedding instead of aborting.

## Fix principle

Make model loading tolerant to equivalent GGUF metadata and tensor naming variants used by this model family, so the loader resolves compatible aliases instead of assuming one exact naming scheme.
