import { Chip, Group, ScrollArea, Skeleton, Stack, Text } from "@mantine/core";
import { PageCard, Source } from "./PageCard";
import { ResultsToolbar, ResultsToolbarProps } from "./ResultsToolbar";
import * as classes from "./ResultsPanel.css";

export type { Source };

/** SentencePiece marks word starts with U+2581; it is noise in the UI. */
const cleanToken = (token: string) => token.replace(/^[\u2581_]+/, "") || token;

export interface ResultsPanelProps extends ResultsToolbarProps {
  sources: Source[];
  loading: boolean;
  queryTokens: string[];
  currentMap: number;
  onCurrentMapChange: (index: number) => void;
  tokenMaps: number[][][][] | null;
}

export function ResultsPanel(props: ResultsPanelProps) {
  const {
    sources,
    loading,
    queryTokens,
    currentMap,
    onCurrentMapChange,
    tokenMaps,
    ...toolbarProps
  } = props;

  return (
    <div className={classes.panel}>
      <ResultsToolbar {...toolbarProps} />

      {queryTokens.length > 0 && (
        <Group gap={6} className={classes.tokens}>
          <Text fz="sm" c="dimmed" pr={4}>
            Attention for
          </Text>
          {queryTokens.map((token, idx) => (
            <Chip
              key={`${token}-${idx}`}
              size="sm"
              radius="sm"
              checked={idx === currentMap}
              onChange={() => onCurrentMapChange(idx)}
            >
              {cleanToken(token)}
            </Chip>
          ))}
        </Group>
      )}

      <ScrollArea className={classes.scroll}>
        <Stack gap="md" p="md" align="flex-start">
          {sources.map((source, idx) => (
            <PageCard
              key={`${source.name}-${source.page}`}
              source={source}
              tokenMap={tokenMaps?.[idx]?.[currentMap] ?? null}
              color={props.color}
              opacity={props.opacity}
              zoom={props.zoom}
            />
          ))}

          {sources.length === 0 &&
            loading &&
            Array.from({ length: 2 }).map((_, idx) => (
              <Skeleton key={idx} height={280} radius="md" w="100%" />
            ))}

          {sources.length === 0 && !loading && (
            <Text fz="md" c="dimmed" ta="center" pt="xl" w="100%">
              The pages your answer is based on will show up here.
            </Text>
          )}
        </Stack>
      </ScrollArea>
    </div>
  );
}
