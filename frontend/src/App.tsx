import "@mantine/core/styles.css";
import "@mantine/dropzone/styles.css";
import "./styles/global.css";
import { MantineProvider } from "@mantine/core";
import { theme } from "./theme";
import Header from "./Header/Header";
import { WelcomeScreen } from "./Welcome/WelcomeScreen";
import { SplitPane } from "./SplitPane/SplitPane";
import { ChatPanel } from "./Chat/ChatPanel";
import { ResultsPanel, Source } from "./Results/ResultsPanel";

import { HeatmapColor, ZOOM_DEFAULT } from "./Results/types";
import { useEffect, useState } from "react";
import * as api from "./api";
import * as classes from "./App.css";

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

type Results = {
  results: Source[];
};

type AttentionMapData = {
  token: string;
  attention_map: number[][];
};

type Heatmaps = {
  query_tokens: string[];
  heatmaps: AttentionMapData[][];
};

export default function App() {
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [uploadStatus, setUploadStatus] = useState<string>("");
  const [collection, setCollection] = useState<string | null>(null);
  const [question, setQuestion] = useState<string | null>(null);
  const [response, setResponse] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [sourceCount, setSourceCount] = useState<number>(3);
  const [heatmaps, setHeatmaps] = useState<Heatmaps | null>(null);
  const [zoom, setZoom] = useState<number>(ZOOM_DEFAULT);
  const [tokenMaps, setTokenMaps] = useState<number[][][][] | null>(null);
  const [currentMap, setCurrentMap] = useState<number>(0);
  const [color, setColor] = useState<HeatmapColor>("red");
  const [opacity, setOpacity] = useState<number>(60);

  const handleQuestionSubmit = async (newQuestion: string) => {
    setQuestion(newQuestion);
    setResponse(null);
    setError(null);
    setLoading(true);
    setSources([]);
    setHeatmaps(null);
    setTokenMaps(null);
    setCurrentMap(0);
    await api.postStream<SearchRequest, SearchResponse | Results | string>(
      "search",
      { query: newQuestion, instance_id: collection ?? "", count: sourceCount },
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
      },
      undefined,
      undefined,
      (err) => setError(err.message)
    );
    setLoading(false);
  };

  const handleFilesUpload = async (files: File[]) => {
    setUploading(true);
    setError(null);
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
      },
      undefined,
      undefined,
      (err) => setError(err.message)
    );
    setUploading(false);
  };

  const handleReset = () => {
    setUploadedFiles([]);
    setCollection(null);
    setQuestion(null);
    setResponse(null);
    setError(null);
    setSources([]);
    setHeatmaps(null);
    setTokenMaps(null);
    setCurrentMap(0);
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
  }, [heatmaps]);

  const hasDocuments = uploadedFiles.length > 0;

  return (
    <MantineProvider theme={theme}>
      <div className={classes.shell}>
        <Header />
        <main className={classes.main}>
          {!hasDocuments ? (
            <WelcomeScreen
              uploading={uploading}
              uploadStatus={uploadStatus}
              error={error}
              onSubmitFiles={handleFilesUpload}
            />
          ) : (
            <SplitPane
              defaultRatio={0.3}
              left={
                <ChatPanel
                  question={question}
                  response={response}
                  error={error}
                  loading={loading}
                  disabled={uploading}
                  onSubmit={handleQuestionSubmit}
                />
              }
              right={
                <ResultsPanel
                  filenames={uploadedFiles}
                  onReset={handleReset}
                  sourceCount={sourceCount}
                  onSourceCountChange={setSourceCount}
                  color={color}
                  onColorChange={setColor}
                  opacity={opacity}
                  onOpacityChange={setOpacity}
                  disabled={loading}
                  sources={sources}
                  loading={loading}
                  queryTokens={heatmaps?.query_tokens ?? []}
                  currentMap={currentMap}
                  onCurrentMapChange={setCurrentMap}
                  tokenMaps={tokenMaps}
                  zoom={zoom}
                  onZoomChange={setZoom}
                />
              }
            />
          )}
        </main>
      </div>
    </MantineProvider>
  );
}
