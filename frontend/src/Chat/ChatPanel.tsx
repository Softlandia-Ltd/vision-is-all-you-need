import { useEffect, useRef } from "react";
import {
  Alert,
  ScrollArea,
  Stack,
  Text,
  TypographyStylesProvider,
} from "@mantine/core";
import { IconAlertTriangle } from "@tabler/icons-react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Question } from "../Question/Question";
import * as classes from "./ChatPanel.css";

export interface ChatPanelProps {
  question: string | null;
  response: string | null;
  error: string | null;
  loading: boolean;
  disabled: boolean;
  onSubmit: (question: string) => void;
}

export function ChatPanel(props: ChatPanelProps) {
  const viewportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    viewportRef.current?.scrollTo({
      top: viewportRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [props.response, props.question]);

  const hasAnswer = props.response !== null && props.response !== "";

  return (
    <div className={classes.panel}>
      <ScrollArea className={classes.transcript} viewportRef={viewportRef}>
        <Stack gap="lg" p="md">
          {!props.question && (
            <Text fz="md" c="dimmed" ta="center" pt="xl">
              Ask a question and the answer will appear here, with the pages it
              came from on the right.
            </Text>
          )}
          {props.question && (
            <div className={classes.bubble}>
              <Text fz="md">{props.question}</Text>
            </div>
          )}
          {props.error && (
            <Alert
              color="red"
              variant="light"
              icon={<IconAlertTriangle size={18} />}
              title="The request failed"
            >
              <Text fz="xs" style={{ overflowWrap: "anywhere" }}>
                {props.error}
              </Text>
            </Alert>
          )}
          {(hasAnswer || (props.loading && !props.error)) && (
            <Stack gap="xs">
              <Text fz="sm" fw={600} c="dimmed" tt="uppercase">
                Answer
              </Text>
              {hasAnswer ? (
                <TypographyStylesProvider className={classes.answer}>
                  <ReactMarkdown
                    children={props.response ?? ""}
                    components={{
                      code(codeProps) {
                        const { children, className, node, ...rest } =
                          codeProps;
                        void node;
                        const match = /language-(\w+)/.exec(className || "");
                        return match ? (
                          <SyntaxHighlighter
                            {...rest}
                            PreTag="div"
                            language={match[1]}
                            style={dark}
                          >
                            {String(children).replace(/\n$/, "")}
                          </SyntaxHighlighter>
                        ) : (
                          <code {...rest} className={className}>
                            {children}
                          </code>
                        );
                      },
                    }}
                  />
                </TypographyStylesProvider>
              ) : (
                <div className={classes.typing} aria-label="Thinking">
                  <span className={classes.dot} />
                  <span className={classes.dot} />
                  <span className={classes.dot} />
                </div>
              )}
            </Stack>
          )}
        </Stack>
      </ScrollArea>
      <div className={classes.composer}>
        <Question
          onSubmit={props.onSubmit}
          loading={props.loading}
          disabled={props.disabled}
        />
      </div>
    </div>
  );
}
