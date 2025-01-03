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
  Image as MantineImage,
  Stack,
  Text,
  ScrollArea,
  Skeleton,
  Flex,
  Slider,
  Chip,
  SegmentedControl,
} from "@mantine/core";
import { theme } from "./theme";
import Header from "./Header/Header";
import { DropzoneBox } from "./Dropzone/Dropzone";
import { Question } from "./Question/Question";
import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { dark } from "react-syntax-highlighter/dist/esm/styles/prism";
import * as api from "./api";
import { IconClearAll } from "@tabler/icons-react";
import * as classes from "./App.css";
import { AttentionMap } from "./AttentionMap/AttentionMap";

type UploadResponse = {
  id: string;
  filenames: string[];
  message: string;
};

type SearchRequest = {
  query: string;
  instance_id: string;
  count: number;
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

type AttentionMap = {
  token: string;
  attention_map: number[][];
};

type Heatmaps = {
  query_tokens: string[];
  heatmaps: AttentionMap[][];
};

export default function App() {
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [uploadStatus, setUploadStatus] = useState<string>("");
  const [collection, setCollection] = useState<string | null>(null);
  const [response, setResponse] = useState<string | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [sourceCount, setSourceCount] = useState<number>(3);
  const [heatmaps, setHeatmaps] = useState<Heatmaps | null>(null);
  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);
  const [tokenMaps, setTokenMaps] = useState<number[][][][] | null>(null);
  const [currentMap, setCurrentMap] = useState<number>(0);
  const [color, setColor] = useState<"red" | "blue" | "green">("red");
  const [opacity, setOpacity] = useState<number>(60);

  const handleQuestionSubmit = async (question: string) => {
    setResponse(null);
    setLoading(true);
    setSources([]);
    setHeatmaps(null);
    setTokenMaps(null);
    setCurrentMap(0);
    await api.postStream<SearchRequest, SearchResponse | Results | string>(
      "search",
      { query: question, instance_id: collection ?? "", count: sourceCount },
      (data, event) => {
        if (event === "sources") {
          setSources((data as Results).results);
        } else if (event === "heatmaps") {
          setHeatmaps(JSON.parse(data as string) as Heatmaps);
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
        if (data.message && data.message !== "") {
          setUploadStatus(data.message);
        }
        if (event === "complete") {
          setUploadedFiles(data.filenames);
          setCollection(data.id);
          setUploadStatus("");
        }
      }
    );
    setUploading(false);
  };

  useEffect(() => {
    if (heatmaps) {
      const newTokenMaps: number[][][][] = [];

      const mapHeight = 32;
      const mapWidth = 32;

      heatmaps.heatmaps.forEach((imageHeatmaps) => {
        const tokenMapsForImage: number[][][] = [];
        const combinedAttentionMap: number[][] = [];

        for (let i = 0; i < mapHeight; i++) {
          combinedAttentionMap.push(new Array(mapWidth).fill(0));
        }

        imageHeatmaps.forEach((heatmap) => {
          const tokenMap: number[][] = [];
          for (let i = 0; i < mapHeight; i++) {
            tokenMap.push(new Array(mapWidth).fill(0));
          }

          heatmap.attention_map.forEach((row, rowIndex) => {
            row.forEach((value, colIndex) => {
              combinedAttentionMap[rowIndex][colIndex] += value;
              tokenMap[rowIndex][colIndex] = value;
            });
          });

          tokenMapsForImage.push(tokenMap);
        });

        newTokenMaps.push(tokenMapsForImage);
      });

      setTokenMaps(newTokenMaps);
    }
  }, [heatmaps, width, height]);

  return (
    <MantineProvider theme={theme}>
      <Stack w="100%" h="100%" p={{ base: "xs", sm: "sm" }}>
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
            <Group w="100%" justify="center" align="center" gap="md">
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
        <Flex
          direction={{ base: "column", sm: "row" }}
          w="100%"
          mah="100%"
          h="100%"
          mih={0}
          justify="space-between"
        >
          <Stack
            h={{ base: "50%", sm: "100%" }}
            mah="100%"
            pr={{ base: "none", sm: "md" }}
            w={{ base: "100%", sm: "48%" }}
            miw={{ base: "100%", sm: "48%" }}
            pb={{ base: "sm", md: "none", xl: "none", lg: "none" }}
          >
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
          <Divider
            display={{ base: "none", sm: "block" }}
            orientation="vertical"
            maw="1px"
          />
          <Divider
            display={{ base: "block", sm: "none" }}
            orientation="horizontal"
            mah="1px"
          />
          <Stack
            h="100%"
            mah="100%"
            w={{ base: "100%", sm: "48%" }}
            miw={{ base: "100%", sm: "48%" }}
            gap="md"
            pl="md"
            pr="md"
            pt={{ base: "xs", md: "none", xl: "none", lg: "none" }}
            className={classes.sources}
          >
            <Stack>
              <Group grow justify="space-between" gap="xl">
                <Stack>
                  <Text fz={{ base: "md", sm: "xl" }}>
                    Using top {sourceCount} matches
                  </Text>
                  <Slider
                    color="orange"
                    defaultValue={3}
                    size={"sm"}
                    step={1}
                    onChange={(value) => setSourceCount(value)}
                    value={sourceCount}
                    min={3}
                    max={10}
                    pb={"xl"}
                    disabled={loading}
                    marks={[
                      { value: 3, label: "3" },
                      { value: 4, label: "4" },
                      { value: 5, label: "5" },
                      { value: 6, label: "6" },
                      { value: 7, label: "7" },
                      { value: 8, label: "8" },
                      { value: 9, label: "9" },
                      { value: 10, label: "10" },
                    ]}
                  />
                </Stack>
                <Stack gap="sm">
                  <Text size="sm" fw={500}>
                    Heatmap color
                  </Text>
                  <SegmentedControl
                    value={color}
                    color={color}
                    size="xs"
                    variant="light"
                    onChange={(val) =>
                      setColor(val as "red" | "blue" | "green")
                    }
                    data={[
                      { label: "Red", value: "red" },
                      { label: "Green", value: "green" },
                      { label: "Blue", value: "blue" },
                    ]}
                  />
                  <Slider
                    color="orange"
                    label="Heatmap opacity"
                    size="xs"
                    value={opacity}
                    onChange={(value) => setOpacity(value)}
                    marks={[
                      { value: 20, label: "20%" },
                      { value: 50, label: "50%" },
                      { value: 80, label: "80%" },
                    ]}
                  />
                </Stack>
              </Group>
              <Group pt="md">
                {heatmaps?.query_tokens.map((token, idx) => (
                  <Chip
                    key={idx}
                    color="grape"
                    size="sm"
                    checked={idx == currentMap}
                    radius="xl"
                    onChange={() => setCurrentMap(idx)}
                  >
                    {token}
                  </Chip>
                ))}
              </Group>
            </Stack>
            {sources.length > 0 && (
              <ScrollArea h="100%" mah="100%">
                {sources.map((source, idx) => (
                  <Card shadow="sm" padding="lg" radius="md" mb="lg" withBorder>
                    <Group justify="space-between" pb="md">
                      <Text fw={500}>
                        {source.name} - Page {source.page}
                      </Text>
                      <Badge color="pink">Score {source.score}</Badge>
                    </Group>
                    <Card.Section style={{ position: "relative" }}>
                      {/* Original Image */}
                      <MantineImage
                        src={"data:image/jpeg;base64," + source.image}
                        height={"100%"}
                        width={"100%"}
                        alt="Original image"
                        onLoad={(img) => {
                          setWidth(img.currentTarget.width);
                          setHeight(img.currentTarget.height);
                        }}
                      />
                      <AttentionMap
                        tokenMap={tokenMaps?.[idx]?.[currentMap] ?? null}
                        width={width}
                        height={height}
                        color={color}
                        opacity={opacity}
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
        </Flex>
      </Stack>
    </MantineProvider>
  );
}
