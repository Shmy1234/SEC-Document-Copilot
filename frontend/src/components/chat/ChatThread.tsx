import { useChat } from '@ai-sdk/react'
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
  type FormEvent,
} from 'react'
import { Link } from 'react-router-dom'

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation'
import {
  PromptInput,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbar,
} from '@/components/ai-elements/prompt-input'
import { AssistantMessage } from '@/components/chat/AssistantMessage'
import { ChatEmptyState } from '@/components/chat/ChatEmptyState'
import { EvidencePanel, type SelectedCitation } from '@/components/chat/EvidencePanel'
import { RunStatus } from '@/components/chat/RunStatus'
import {
  describeError,
  describeStreamError,
  type ErrorDescription,
} from '@/components/chat/chat-errors'
import { messageErrorCode, messageStages, messageText } from '@/components/chat/message-format'
import { ErrorNotice } from '@/components/common/ErrorNotice'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { chatApi } from '@/lib/chat-api'
import { chatRegistry } from '@/lib/chat-registry'
import { deriveThreadTitle } from '@/lib/format'

type ChatThreadProps = {
  threadId: string
  hasTitle: boolean
  onThreadChanged: () => void | Promise<void>
}

export function ChatThread({ threadId, hasTitle, onThreadChanged }: ChatThreadProps) {
  const [input, setInput] = useState('')
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [historyError, setHistoryError] = useState<ErrorDescription | null>(null)
  const [selected, setSelected] = useState<SelectedCitation | null>(null)
  const [historyToken, setHistoryToken] = useState(0)
  const lastChipRef = useRef<HTMLButtonElement | null>(null)
  const composerRef = useRef<HTMLTextAreaElement>(null)

  // The instance outlives this component, so a run started here keeps streaming
  // while another thread is on screen and is rejoined mid-flight on return.
  const chat = chatRegistry.chatFor(threadId)
  const { messages, status, stop } = useChat({ chat })

  // A send that fails after the reader has moved on still belongs to its thread,
  // so the failure lives in the registry rather than in this component's state.
  useSyncExternalStore(chatRegistry.subscribe, chatRegistry.version)
  const sendFailure = chatRegistry.sendFailure(threadId)

  useEffect(() => {
    let cancelled = false

    async function loadMessages() {
      // Returning to a thread already in memory must not blank it behind a
      // skeleton — least of all while its answer is still streaming in.
      setLoadingHistory(!chatRegistry.isLoaded(threadId))
      setHistoryError(null)
      try {
        await (historyToken === 0
          ? chatRegistry.loadHistory(threadId)
          : chatRegistry.reloadHistory(threadId))
      } catch (unknownError) {
        if (!cancelled) setHistoryError(describeError(unknownError))
      } finally {
        if (!cancelled) setLoadingHistory(false)
      }
    }

    void loadMessages()
    return () => {
      cancelled = true
    }
  }, [threadId, historyToken])

  const busy = status === 'streaming' || status === 'submitted'

  async function send(text: string) {
    setInput('')
    await chatRegistry.send(threadId, text)
    if (chatRegistry.sendFailure(threadId)) return
    try {
      // The thread is named after the question that started it.
      if (!hasTitle) await chatApi.updateThreadTitle(threadId, deriveThreadTitle(text))
      await onThreadChanged()
    } catch {
      // The answer is saved; an unnamed thread in the sidebar is not worth
      // reporting as a failed turn.
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const text = input.trim()
    if (!text || busy) return
    void send(text)
  }

  /** Closing the rail returns focus to the chip that opened it. */
  const closeEvidence = useCallback(() => {
    setSelected(null)
    lastChipRef.current?.focus()
    lastChipRef.current = null
  }, [])

  const selectCitation = useCallback(
    (messageId: string, citation: { citation_index: number }, chip: HTMLButtonElement | null) => {
      lastChipRef.current = chip
      setSelected({ messageId, citationIndex: citation.citation_index })
    },
    [],
  )

  const lastMessage = messages.at(-1)
  // Turn failures render beside the turn that failed; only transport and history
  // problems belong at the bottom of the thread.
  const activeError = sendFailure ? describeError(sendFailure.error) : historyError
  const stages = busy && lastMessage?.role === 'assistant' ? messageStages(lastMessage) : []

  function retryLastSend() {
    if (!sendFailure) return
    chatRegistry.clearSendFailure(threadId)
    void send(sendFailure.text)
  }

  /** The question a failed turn was answering, so it can be asked again. */
  function questionBefore(index: number): string | null {
    const previous = messages[index - 1]
    if (previous?.role !== 'user') return null
    return messageText(previous).trim() || null
  }

  return (
    <div className="flex min-h-0 flex-1">
      <div className="flex min-w-0 flex-1 flex-col">
        <Conversation>
          <ConversationContent className="mx-auto w-full max-w-3xl px-4 py-6">
            {loadingHistory ? (
              <div aria-label="Loading messages" className="space-y-6" role="status">
                <Skeleton className="ml-auto h-9 w-2/5 rounded-xl" />
                <div className="space-y-2">
                  <Skeleton className="h-3 w-16" />
                  <Skeleton className="h-3 w-full" />
                  <Skeleton className="h-3 w-11/12" />
                  <Skeleton className="h-3 w-3/4" />
                </div>
              </div>
            ) : messages.length === 0 && !historyError ? (
              <ChatEmptyState
                onPick={(question) => {
                  setInput(question)
                  composerRef.current?.focus()
                }}
              />
            ) : (
              <div className="space-y-6">
                {messages.map((message, index) => {
                  if (message.role !== 'assistant') {
                    return (
                      <div className="flex justify-end" key={message.id}>
                        <div className="max-w-[75%] rounded-xl bg-primary px-3.5 py-2.5 text-primary-foreground">
                          {messageText(message)}
                        </div>
                      </div>
                    )
                  }

                  const errorCode = messageErrorCode(message)
                  const failure = errorCode ? describeStreamError(errorCode) : null
                  const question = failure ? questionBefore(index) : null
                  return (
                    <div className="space-y-3" key={message.id}>
                      {messageText(message) ? (
                        <AssistantMessage
                          message={message}
                          onCitationSelect={selectCitation}
                          selectedCitationIndex={
                            selected?.messageId === message.id ? selected.citationIndex : null
                          }
                          streaming={busy && index === messages.length - 1}
                        />
                      ) : null}
                      {failure ? (
                        <ErrorNotice
                          description={failure.description}
                          onRetry={
                            question && !busy ? () => void send(question) : undefined
                          }
                          title={failure.title}
                          tone={failure.tone}
                        />
                      ) : null}
                    </div>
                  )
                })}

                {busy ? <RunStatus stages={stages} /> : null}

                {activeError ? (
                  <ErrorNotice
                    action={
                      activeError.needsSignIn ? (
                        <Button render={<Link to="/sign-in" />} size="sm" variant="outline">
                          Sign in
                        </Button>
                      ) : null
                    }
                    description={activeError.description}
                    onRetry={
                      !activeError.canRetry
                        ? undefined
                        : sendFailure
                          ? retryLastSend
                          : historyError
                            ? () => setHistoryToken((token) => token + 1)
                            : undefined
                    }
                    title={activeError.title}
                    tone={activeError.tone}
                  />
                ) : null}
              </div>
            )}
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>

        <div className="mx-auto w-full max-w-3xl px-4 pb-4">
          <PromptInput onSubmit={submit}>
            <PromptInputTextarea
              aria-label="Ask about a filing"
              disabled={busy}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask about a filing…"
              ref={composerRef}
              value={input}
            />
            <PromptInputToolbar>
              <span className="pl-1.5 text-[0.6875rem] text-muted-foreground">
                Answers cite the filings they come from
              </span>
              <PromptInputSubmit
                disabled={!busy && !input.trim()}
                onStop={stop}
                status={busy ? 'streaming' : 'ready'}
              />
            </PromptInputToolbar>
          </PromptInput>
        </div>
      </div>

      {selected ? <EvidencePanel onClose={closeEvidence} selection={selected} /> : null}
    </div>
  )
}
