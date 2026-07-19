import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'

// ponytail: hardcoded, no env config until there's a second environment to point at
const API = 'http://localhost:8000'

type Source = { file_path: string; start_line: number; github_url: string | null }
type EmbedResult = { files_embedded: number; total_chunks_in_collection: number }

export default function App() {
  const [githubUrl, setGithubUrl] = useState('')
  const [branch, setBranch] = useState('main')
  const [embedBusy, setEmbedBusy] = useState(false)
  const [embedError, setEmbedError] = useState('')
  const [embedResult, setEmbedResult] = useState<EmbedResult | null>(null)
  // Set from a successful /embed_repo call and sent as /ask's repo_url —
  // simplest way to scope questions to "the repo I just embedded" without
  // a repo picker. null means no repo embedded yet this session, so /ask
  // omits repo_url and searches everything, matching the backend's
  // existing unscoped-query default.
  const [embeddedRepoUrl, setEmbeddedRepoUrl] = useState<string | null>(null)

  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState<Source[]>([])
  const [busy, setBusy] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState('')

  async function embedRepo(e: React.FormEvent) {
    e.preventDefault()
    setEmbedBusy(true)
    setEmbedError('')
    setEmbedResult(null)

    try {
      const res = await fetch(`${API}/embed_repo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ github_url: githubUrl, branch }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail ?? `request failed: ${res.status}`)
      setEmbedResult(data)
      setEmbeddedRepoUrl(githubUrl)
    } catch (err) {
      setEmbedError(err instanceof Error ? err.message : String(err))
    } finally {
      setEmbedBusy(false)
    }
  }

  async function ask(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setStreaming(true)
    setAnswer('')
    setSources([])
    setError('')

    try {
      const res = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          ...(embeddedRepoUrl ? { repo_url: embeddedRepoUrl } : {}),
        }),
      })
      if (!res.ok || !res.body) throw new Error(`request failed: ${res.status}`)

      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += value

        // SSE frames are separated by a blank line; parse each complete one
        let boundary
        while ((boundary = buffer.indexOf('\n\n')) !== -1) {
          const frame = buffer.slice(0, boundary)
          buffer = buffer.slice(boundary + 2)

          const eventLine = frame.split('\n').find((l) => l.startsWith('event:'))
          const dataLine = frame.split('\n').find((l) => l.startsWith('data:'))
          if (!eventLine || !dataLine) continue

          const event = eventLine.slice('event:'.length).trim()
          const data = JSON.parse(dataLine.slice('data:'.length).trim())

          if (event === 'token') setAnswer((prev) => prev + data)
          else if (event === 'sources') {
            setSources(data)
            setStreaming(false)
          } else if (event === 'error') {
            setError(data)
            setStreaming(false)
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
      setStreaming(false)
    }
  }

  return (
    <main>
      <div className="panel">
        <div className="panel-header">
          <span className="mark">❯</span>
          <h1 className="title">chat with your repo</h1>
        </div>

        <div className="panel-body">
          <p className="section-label">Index a repo</p>
          <form onSubmit={embedRepo}>
            <div>
              <label htmlFor="github_url">GitHub repo URL</label>
              <input
                id="github_url"
                value={githubUrl}
                onChange={(e) => setGithubUrl(e.target.value)}
                placeholder="https://github.com/owner/repo"
                required
              />
            </div>
            <div>
              <label htmlFor="branch">Branch</label>
              <input
                id="branch"
                value={branch}
                onChange={(e) => setBranch(e.target.value)}
                placeholder="main"
                required
              />
            </div>
            <button type="submit" disabled={embedBusy}>
              {embedBusy ? 'Embedding…' : 'Embed Repo'}
            </button>
          </form>

          {embedError && <p className="error">{embedError}</p>}

          {embedResult && (
            <p className="success">
              Embedded {embedResult.files_embedded} file
              {embedResult.files_embedded === 1 ? '' : 's'} — {embedResult.total_chunks_in_collection}{' '}
              chunks total in the collection.
            </p>
          )}

          <div className="section-divider">
            <p className="section-label">
              Ask a question{embeddedRepoUrl ? ` — scoped to ${embeddedRepoUrl}` : ''}
            </p>
            <form onSubmit={ask}>
              <div>
                <label htmlFor="question">question</label>
                <input
                  id="question"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="what does get_chunks return?"
                  required
                />
              </div>
              <button type="submit" disabled={busy}>
                {busy ? 'Asking…' : 'Ask'}
              </button>
            </form>
          </div>

          {error && <p className="error">{error}</p>}

          {answer && (
            <div className="answer-card">
              {/* A <div>, not <p> — ReactMarkdown can render block elements
                  (pre, ul, headings) as children, which aren't valid inside
                  a <p>. The cursor is a sibling AFTER the markdown tree, not
                  inside it, so a still-open ``` fence mid-stream can never
                  swallow it — react-markdown just renders unclosed markdown
                  as literal text until the fence closes; it doesn't throw. */}
              <div className="answer">
                <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{answer}</ReactMarkdown>
                {streaming && <span className="cursor">▍</span>}
              </div>
            </div>
          )}

          {sources.length > 0 && (
            <div className="section-divider">
              <p className="section-label">Sources</p>
              <ul className="sources">
                {sources.map((s, i) => {
                  const label = `${s.file_path}:${s.start_line}`
                  return (
                    <li key={i}>
                      {s.github_url ? (
                        <a
                          className="source-chip"
                          href={s.github_url}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          {label}
                        </a>
                      ) : (
                        <span className="source-chip">{label}</span>
                      )}
                    </li>
                  )
                })}
              </ul>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
