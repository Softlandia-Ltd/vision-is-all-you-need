import { useState } from "react";
import { Badge, Card, Group, Image as MantineImage, Text } from "@mantine/core";
import { AttentionMap } from "../AttentionMap/AttentionMap";
import { HeatmapColor } from "./types";
import * as classes from "./ResultsPanel.css";

export type Source = {
  score: number;
  image: string;
  page: number;
  name: string;
};

export interface PageCardProps {
  source: Source;
  tokenMap: number[][] | null;
  color: HeatmapColor;
  opacity: number;
  zoom: number;
}

export function PageCard({
  source,
  tokenMap,
  color,
  opacity,
  zoom,
}: PageCardProps) {
  // Natural pixel size drives the canvas resolution; it never changes with zoom.
  const [size, setSize] = useState({ width: 0, height: 0 });

  return (
    <Card
      shadow="xs"
      padding="md"
      withBorder
      className={classes.pageCard}
      style={{ width: `${zoom * 100}%` }}
    >
      <Group justify="space-between" pb="sm" wrap="nowrap">
        <Text fz="md" fw={600} truncate>
          {source.name} · page {source.page}
        </Text>
        <Badge variant="light" color="gray" size="md">
          {source.score.toFixed(3)}
        </Badge>
      </Group>
      <Card.Section style={{ position: "relative" }}>
        <MantineImage
          src={"data:image/jpeg;base64," + source.image}
          w="100%"
          h="auto"
          alt={`${source.name} page ${source.page}`}
          onLoad={(event) =>
            setSize({
              width: event.currentTarget.naturalWidth,
              height: event.currentTarget.naturalHeight,
            })
          }
        />
        <AttentionMap
          tokenMap={tokenMap}
          width={size.width}
          height={size.height}
          color={color}
          opacity={opacity}
        />
      </Card.Section>
    </Card>
  );
}
