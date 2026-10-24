# Cloud Voice

Cloud Voice turns TuneLab notes into audio by sending them to a **cloud singing-voice synthesis
service over HTTP(S)** — or to its built-in offline demo engine when no service is configured yet.

## What it does

- Reads the part's notes (lyrics, notes, pinned phonemes), the pitch track, pitch deviation and the
  `dynamics` / `breathiness` automation tracks.
- Samples that musical information into a single JSON request (protocol `tunelab-cloudvoice` v1)
  and POSTs it to your configured endpoint.
- Pulls the provider's voicebank catalog automatically from `GET <service base>/voices` on load —
  you never paste a voice list; only the service URL and API key are configured.
- Reads the backend capability schema (`/voices.schema` or `GET <service base>/schema`) and builds the
  part/note property panels, editable automation lanes, read-only echo lanes and auto-pitch support from it.
- Decodes the returned audio (base64 PCM / WAV, or a download URL), then publishes it together with
  synthesized phonemes, a pitch echo and an energy parameter echo.
- Splits the part by note gaps, synthesizes one block at a time, and re-synthesizes only the blocks
  affected by an edit (move/resize/lyric/pitch/curve changes).

## Getting started

1. Open **Settings → Extensions → Cloud Voice**.
2. Fill in the **Service URL** (the `.../synthesize` endpoint) and the **API Key**. That is all:
   the plugin pulls the provider's voicebank catalog from `GET .../voices` automatically on load.
   Leave the service URL empty to use the built-in offline demo voicebanks.
3. Create a **voice part**, choose a voice bank from the provider's list.
4. Type lyrics (one Chinese character or Latin syllable like `la` per note), then play or edit —
   synthesis runs in the background with a status band per block.

## Notes

- The engine id `CloudVoice` is stored in project files. The settings are per extension and shared
  across projects; the API key is stored in the platform keychain by the host, never in project files.
- Lyrics with `-` act as **continuation** notes: they extend the previous note's vowel (melisma)
  instead of starting a new syllable, matching the editor's standard convention.
- If the backend declares `pitch.auto`, the plugin requests automatic pitch and shows the F0 the
  backend actually rendered as the synthesized pitch curve.
- If the cloud returns an error, that block turns red with the server message in the status band;
  edit the block (or fix the settings) to retry.