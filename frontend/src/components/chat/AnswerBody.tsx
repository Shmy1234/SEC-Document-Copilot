import { memo, type ReactNode } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

type AnswerBodyProps = {
  text: string
  streaming: boolean
}

/**
 * Answers are markdown because filing answers are frequently tabular. Rendered to
 * React elements rather than HTML, so model output can never inject markup.
 */
const remarkPlugins = [remarkGfm]
const components = {
  table: ({ children }: { children?: ReactNode }) => (
    <div className="my-3 overflow-x-auto rounded-md border">
      <table>{children}</table>
    </div>
  ),
}

/** Memoised so markdown is re-parsed only when this answer's text changes. */
export const AnswerBody = memo(function AnswerBody({ text, streaming }: AnswerBodyProps) {
  return (
    <div className="answer-prose">
      <Markdown components={components} remarkPlugins={remarkPlugins}>
        {text}
      </Markdown>
      {streaming ? (
        <span
          aria-hidden
          className="ml-0.5 inline-block h-3.5 w-[2px] translate-y-0.5 animate-pulse bg-foreground"
        />
      ) : null}
    </div>
  )
})
