import "@mantine/core/styles.css";
import "@mantine/dropzone/styles.css";
import "./styles/global.css";
import {
  Badge,
  Box,
  Button,
  Card,
  Divider,
  Group,
  Loader,
  LoadingOverlay,
  MantineProvider,
  Image,
  Stack,
  Text,
  ScrollArea,
  Skeleton,
} from "@mantine/core";
import { theme } from "./theme";
import Header from "./Header/Header";
import { DropzoneBox } from "./Dropzone/Dropzone";
import { Question } from "./Question/Question";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dark } from "react-syntax-highlighter/dist/esm/styles/prism";
import * as api from "./api";
import { IconClearAll } from "@tabler/icons-react";
import * as classes from "./App.css";

type UploadResponse = {
  id: string;
  filenames: string[];
  message: string;
};

type SearchRequest = {
  query: string;
  instance_id: string;
};

type SearchResponse = {
  chunk: string;
};

type Source = {
  score: number;
  image: string;
  page: number;
  name: string;
};

type Results = {
  results: Source[];
};

export default function App() {
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [uploadStatus, setUploadStatus] = useState<string>("");
  const [collection, setCollection] = useState<string | null>(null);
  const [response, setResponse] = useState<string | null>(null);
  const [sources, setSources] = useState<Source[]>([]);

  const handleQuestionSubmit = async (question: string) => {
    setResponse(null);
    setLoading(true);
    setSources([]);
    await api.postStream<SearchRequest, SearchResponse | Results>(
      "search",
      { query: question, instance_id: collection ?? "" },
      (data, event) => {
        if (event === "sources") {
          setSources((data as Results).results);
        } else {
          setResponse((old) => {
            const res = data as SearchResponse;
            return old ? `${old}${res.chunk}` : res.chunk;
          });
        }
      }
    );
    setLoading(false);
  };

  const handleFilesUpload = async (files: File[]) => {
    setUploading(true);
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file, file.name));
    await api.postFilesStream<UploadResponse>(
      "collections",
      formData,
      (data, event) => {
        if (data.message !== "") {
          setUploadStatus(data.message);
        }
        if (event === "complete") {
          setUploadedFiles(data.filenames);
          setCollection(data.id);
        }
      }
    );
    setUploading(false);
  };

  return (
    <MantineProvider theme={theme}>
      <Stack w="100%" h="100%" p="sm">
        <Header />
        <Group grow w="100%" justify="center" mah={"100%"}>
          <Box pos="relative">
            <LoadingOverlay
              visible={uploading}
              zIndex={1000}
              overlayProps={{ radius: "sm", blur: 2 }}
              loaderProps={{
                children: (
                  <Stack gap="md" justify="center" align="center">
                    <Loader color="pink" />
                    <Text>{uploadStatus}</Text>
                  </Stack>
                ),
              }}
            />
            {uploadedFiles.length === 0 && (
              <DropzoneBox
                uploading={uploading}
                onSubmit={(files) => handleFilesUpload(files)}
              />
            )}
            <Group w="100%" justify="center" align="center" gap="xl">
              {uploadedFiles.map((filename) => (
                <Badge
                  key={filename}
                  size="xl"
                  variant="gradient"
                  gradient={{ from: "blue", to: "red", deg: 88 }}
                >
                  {filename}
                </Badge>
              ))}
              {uploadedFiles.length > 0 && (
                <Button
                  color="orange"
                  radius="xl"
                  variant="light"
                  rightSection={<IconClearAll />}
                  onClick={() => {
                    setUploadedFiles([]);
                    setCollection(null);
                  }}
                >
                  Reset
                </Button>
              )}
            </Group>
          </Box>
        </Group>
        <Divider />
        <Group grow w="100%" mah="100%" h="100%" mih={0} justify="center">
          <Stack h="100%" mah="100%" pr="md" w="48%" miw="48%">
            <Question
              onSubmit={handleQuestionSubmit}
              loading={loading}
              disabled={uploading || uploadedFiles.length === 0}
            />
            {response && response !== "" && (
              <ScrollArea h="100%" mah="100%">
                <ReactMarkdown
                  children={response}
                  components={{
                    code(props) {
                      const { children, className, node, ...rest } = props;
                      const match = /language-(\w+)/.exec(className || "");
                      return match ? (
                        <SyntaxHighlighter
                          {...rest}
                          PreTag="div"
                          children={String(children).replace(/\n$/, "")}
                          language={match[1]}
                          style={dark}
                        />
                      ) : (
                        <code {...rest} className={className}>
                          {children}
                        </code>
                      );
                    },
                  }}
                />
              </ScrollArea>
            )}
          </Stack>
          <Divider orientation="vertical" maw="1px" />
          <Stack
            h="100%"
            mah="100%"
            w="48%"
            miw="48%"
            gap="md"
            pl="md"
            pr="md"
            className={classes.sources}
          >
            <Text size="xl">
              Top 3 sources for the LLM that matched the question
            </Text>
            {sources.length > 0 && (
              <ScrollArea h="100%" mah="100%">
                {sources.map((source) => (
                  <Card shadow="sm" padding="lg" radius="md" mb="lg" withBorder>
                    <Group justify="space-between" pb="md">
                      <Text fw={500}>
                        {source.name} - Page {source.page}
                      </Text>
                      <Badge color="pink">Score {source.score}</Badge>
                    </Group>
                    <Card.Section>
                      <Image
                        src={"data:image/jpeg;base64," + source.image}
                        height={"100%"}
                      />
                    </Card.Section>
                  </Card>
                ))}
              </ScrollArea>
            )}
            {sources.length === 0 && loading && (
              <Skeleton height={"100%"} width={"100%"} visible={loading} />
            )}
          </Stack>
        </Group>
      </Stack>
    </MantineProvider>
  );
}
