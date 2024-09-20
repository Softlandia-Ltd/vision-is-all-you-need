import { Textarea, ActionIcon, useMantineTheme, rem } from "@mantine/core";
import { IconArrowRight } from "@tabler/icons-react";
import { useState } from "react";
import * as classes from "./Question.css";


export interface QuestionProps {
  onSubmit: (question: string) => void;
  loading: boolean;
  disabled: boolean;
}

export function Question(props: QuestionProps) {
  const theme = useMantineTheme();
  const [question, setQuestion] = useState("");

  return (
    <Textarea
      radius="md"
      disabled={props.loading || props.disabled}
      w="100%"
      placeholder="Ask a question"
      value={question}
      onChange={(event) => setQuestion(event.currentTarget.value)}
      rightSectionWidth={82}
      classNames={{
        input: classes.questionInput
      }}
      rightSection={
        <ActionIcon
          size={62}
          radius="md"
          disabled={props.loading || props.disabled}
          loading={props.loading}
          color={theme.primaryColor}
          variant="filled"
          onClick={() => props.onSubmit(question)}
        >
          <IconArrowRight
            style={{ width: rem(18), height: rem(18) }}
            stroke={1.5}
          />
        </ActionIcon>
      }
    />
  );
}
