# Issue draft

## Summary

The HTTP embedding endpoints break on valid JSON when the input strings contain escaped quotes, brackets, or other escaped characters.

## Reproduce

Start the server, then send a batch request with valid escaped content:

```powershell
curl.exe -s http://127.0.0.1:18889/v1/embeddings `
  -H "Content-Type: application/json" `
  -d "{\"input\":[\"plain text\",\"text with ] bracket\",\"text with \\\"quoted\\\" part\",\"line\\nbreak\"]}"
```

You can also hit the Ollama-compatible route:

```powershell
curl.exe -s http://127.0.0.1:18889/api/embed `
  -H "Content-Type: application/json" `
  -d "{\"input\":[\"plain text\",\"text with ] bracket\",\"text with \\\"quoted\\\" part\",\"line\\nbreak\"]}"
```

## Expected

The server returns exactly four embeddings, one per input string.

## Actual

Parsing can split or truncate values incorrectly, which leads to wrong input cardinality and downstream errors such as:

```text
returned 7 embeddings for 6 inputs
```

## Minimum proof

- Before: the request above can produce mismatched counts or malformed parsing when the payload contains valid escaped JSON content.
- After: the same request returns one embedding per provided input string, with no count mismatch.

## Fix principle

Parse JSON string values according to JSON escaping rules instead of extracting substrings with delimiter searches, so valid escaped payloads are treated as structured JSON rather than raw text slices.
