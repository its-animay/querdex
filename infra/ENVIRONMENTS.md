# Environment Configs

## Development
- `QUERDEX_DB=./index_store/querdex.db`
- `QUERDEX_OCR_ENABLED=false`
- `QUERDEX_OCR_PROVIDER=tesseract`

## Staging
- `QUERDEX_DB=/var/lib/querdex/staging.db`
- `QUERDEX_OCR_ENABLED=true`
- `QUERDEX_OCR_PROVIDER=tesseract`
- `QUERDEX_TESSERACT_CMD=tesseract`

## Production
- `QUERDEX_DB=/var/lib/querdex/prod.db`
- `QUERDEX_OCR_ENABLED=true`
- `QUERDEX_OCR_PROVIDER=cloud`
- `QUERDEX_OCR_ENDPOINT=https://ocr.example.com/v1/ocr`
- `QUERDEX_OCR_API_KEY=<secret>`
- run with external backup for `index_store` and database snapshots
