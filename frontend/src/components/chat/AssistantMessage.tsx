import type { UIMessage } from '@ai-sdk/react'
import { Check, Copy, SearchX } from 'lucide-react'
import { memo, useRef, useState } from 'react'

import { AnswerBody } from '@/components/chat/AnswerBody'
import { CitationChip } from '@/components/chat/CitationChip'
import { messageCitations, messageRoute, messageText } from '@/components/chat/message-format'
import { Button } from '@/components/ui/button'
import { isGroundedRoute, type Citation } from '@/lib/chat-api'

type AssistantMessageProps = {
  message: UIMessage
  streaming: boolean
  selectedCitationIndex: number | null
  onCitationSelect: (messageId: string, citation: Citation, chip: HTMLButtonElement | null) => void
}

/** Memoised: while one answer streams, settled answers skip re-rendering. */
export const AssistantMessage = memo(function AssistantMessage({
  message,
  streaming,
  selectedCitationIndex,
  onCitationSelect,
}: AssistantMessageProps) {
  const text = messageText(message)
  const citations = messageCitations(message)
  const chipRefs = useRef(new Map<number, HTMLButtonElement | null>())
  const [copied, setCopied] = useState(false)

  // A finished corpus answer with no citations is the grounding contract refusing
  // to guess. Say so plainly rather than letting it pass as a sourced answer.
  // Instant and direct replies never search the corpus, so the absence of
  // citations says nothing about the evidence — warning there was a false alarm.
  const insufficientEvidence =
    !streaming &&
    text.length > 0 &&
    citations.length === 0 &&
    isGroundedRoute(messageRoute(message))

  async function copyAnswer() {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    globalThis.setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="group/message w-full">
      <p className="mb-1.5 font-mono text-[0.6875rem] tracking-tight text-muted-foreground">
        COPILOT
      </p>

      {insufficientEvidence ? (
        <div className="mb-3 flex items-center gap-2 rounded-md border bg-muted/40 px-2.5 py-1.5 text-muted-foreground">
          <SearchX aria-hidden className="size-3.5 shrink-0" />
          <span className="text-[0.8125rem]">No supporting passage found in the corpus</span>
        </div>
      ) : null}

      <AnswerBody streaming={streaming} text={text} />

      {citations.length ? (
        <div className="mt-4">
          <p className="mb-1.5 font-mono text-[0.6875rem] tracking-tight text-muted-foreground">
            SOURCES
          </p>
          <div className="flex flex-wrap gap-1.5">
            {citations.map((citation) => (
              <CitationChip
                citation={citation}
                key={`${citation.chunk_id}-${citation.citation_index}`}
                onSelect={() =>
                  onCitationSelect(
                    message.id,
                    citation,
                    chipRefs.current.get(citation.citation_index) ?? null,
                  )
                }
                ref={(node) => {
                  chipRefs.current.set(citation.citation_index, node)
                }}
                selected={selectedCitationIndex === citation.citation_index}
              />
            ))}
          </div>
        </div>
      ) : null}

      {!streaming && text ? (
        <div className="mt-2 opacity-0 transition-opacity group-hover/message:opacity-100 focus-within:opacity-100">
          <Button
            aria-label={copied ? 'Answer copied' : 'Copy answer'}
            onClick={() => void copyAnswer()}
            size="icon-xs"
            variant="ghost"
          >
            {copied ? <Check /> : <Copy />}
          </Button>
        </div>
      ) : null}
    </div>
  )
})
