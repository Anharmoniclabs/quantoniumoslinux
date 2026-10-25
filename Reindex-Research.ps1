$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & python tools\index_archive.py
    if ($LASTEXITCODE -ne 0) { throw 'Archive indexing failed.' }
    & python tools\import_conversations.py
    if ($LASTEXITCODE -ne 0) { throw 'Conversation indexing failed.' }
    & python tools\index_native_sources.py
    if ($LASTEXITCODE -ne 0) { throw 'Native source indexing failed.' }
} finally { Pop-Location }
