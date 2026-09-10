import { Textarea, ActionIcon, Group } from "@mantine/core";
import { IconArrowUp } from "@tabler/icons-react";
import { useRef, useState } from "react";
import cx from "clsx";
import * as classes from "./Question.css";

export interface QuestionProps {
  onSubmit: (question: string) => void;
  loading: boolean;
  disabled: boolean;
  size?: "md" | "lg";
  placeholder?: string;
  /** Called when the user clicks the composer while it is disabled. */
  onDisabledClick?: () => void;
}

export function Question(props: QuestionProps) {
  const { size = "md", loading, disabled } = props;
  const [question, setQuestion] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const locked = loading || disabled;
  const canSubmit = !locked && question.trim() !== "";

  const submit = () => {
    if (!canSubmit) return;
    props.onSubmit(question.trim());
    setQuestion("");
  };

  return (
    <div
      className={cx(classes.shell, size === "lg" && classes.shellLg)}
      data-disabled={locked || undefined}
      onClick={() => {
        if (locked) props.onDisabledClick?.();
        else inputRef.current?.focus();
      }}
    >
      <Textarea
        ref={inputRef}
        variant="unstyled"
        autosize
        minRows={size === "lg" ? 2 : 1}
        maxRows={6}
        disabled={locked}
        w="100%"
        placeholder={props.placeholder ?? "Ask a question about your PDFs"}
        value={question}
        onChange={(event) => setQuestion(event.currentTarget.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            submit();
          }
        }}
        classNames={{
          input: cx(classes.input, size === "lg" && classes.inputLg),
        }}
      />
      <Group justify="flex-end">
        <ActionIcon
          size={size === "lg" ? 38 : 32}
          radius="xl"
          disabled={!canSubmit}
          loading={loading}
          variant="filled"
          aria-label="Send question"
          onClick={(event) => {
            event.stopPropagation();
            submit();
          }}
        >
          <IconArrowUp size={18} stroke={2} />
        </ActionIcon>
      </Group>
    </div>
  );
}
