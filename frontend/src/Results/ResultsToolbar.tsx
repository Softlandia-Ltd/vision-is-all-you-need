import {
  Group,
  NumberInput,
  Slider,
  Text,
  Button,
  Tooltip,
  ActionIcon,
  UnstyledButton,
} from "@mantine/core";
import {
  IconRotate,
  IconZoomIn,
  IconZoomOut,
  IconZoomReset,
} from "@tabler/icons-react";
import { FileChip } from "../Files/FileChip";
import {
  HeatmapColor,
  ZOOM_DEFAULT,
  ZOOM_MAX,
  ZOOM_MIN,
  ZOOM_STEP,
} from "./types";
import * as classes from "./ResultsPanel.css";

const COLORS: { value: HeatmapColor; label: string; swatch: string }[] = [
  { value: "red", label: "Red", swatch: "var(--mantine-color-red-6)" },
  { value: "green", label: "Green", swatch: "var(--mantine-color-green-6)" },
  { value: "blue", label: "Blue", swatch: "var(--mantine-color-blue-6)" },
];

export interface ResultsToolbarProps {
  filenames: string[];
  onReset: () => void;
  sourceCount: number;
  onSourceCountChange: (value: number) => void;
  color: HeatmapColor;
  onColorChange: (value: HeatmapColor) => void;
  opacity: number;
  onOpacityChange: (value: number) => void;
  zoom: number;
  onZoomChange: (value: number) => void;
  disabled: boolean;
}

export function ResultsToolbar(props: ResultsToolbarProps) {
  const setZoom = (value: number) =>
    props.onZoomChange(
      Math.min(Math.max(Number(value.toFixed(2)), ZOOM_MIN), ZOOM_MAX)
    );

  return (
    <Group
      className={classes.toolbar}
      justify="space-between"
      wrap="wrap"
      gap="sm"
    >
      <Group gap="xs" wrap="wrap">
        {props.filenames.map((filename) => (
          <FileChip key={filename} name={filename} />
        ))}
        <Button
          size="compact-sm"
          variant="subtle"
          color="gray"
          leftSection={<IconRotate size={16} stroke={1.6} />}
          onClick={props.onReset}
        >
          Reset
        </Button>
      </Group>

      <Group gap="lg" wrap="wrap">
        <Group gap={2} wrap="nowrap">
          <Tooltip label="Zoom out" withArrow>
            <ActionIcon
              variant="subtle"
              color="gray"
              aria-label="Zoom out"
              disabled={props.zoom <= ZOOM_MIN}
              onClick={() => setZoom(props.zoom - ZOOM_STEP)}
            >
              <IconZoomOut size={18} stroke={1.6} />
            </ActionIcon>
          </Tooltip>
          <Tooltip label="Reset zoom" withArrow>
            <UnstyledButton
              className={classes.zoomValue}
              aria-label="Reset zoom"
              onClick={() => setZoom(ZOOM_DEFAULT)}
            >
              {Math.round(props.zoom * 100)}%
            </UnstyledButton>
          </Tooltip>
          <Tooltip label="Zoom in" withArrow>
            <ActionIcon
              variant="subtle"
              color="gray"
              aria-label="Zoom in"
              disabled={props.zoom >= ZOOM_MAX}
              onClick={() => setZoom(props.zoom + ZOOM_STEP)}
            >
              <IconZoomIn size={18} stroke={1.6} />
            </ActionIcon>
          </Tooltip>
          <Tooltip label="Fit to panel" withArrow>
            <ActionIcon
              variant="subtle"
              color="gray"
              aria-label="Fit to panel"
              onClick={() => setZoom(ZOOM_DEFAULT)}
            >
              <IconZoomReset size={18} stroke={1.6} />
            </ActionIcon>
          </Tooltip>
        </Group>

        <Tooltip label="How many pages to retrieve" withArrow openDelay={400}>
          <NumberInput
            size="sm"
            w={96}
            min={3}
            max={10}
            step={1}
            clampBehavior="strict"
            allowDecimal={false}
            aria-label="Pages to retrieve"
            leftSection={
              <Text fz="sm" c="dimmed">
                #
              </Text>
            }
            disabled={props.disabled}
            value={props.sourceCount}
            onChange={(value) =>
              props.onSourceCountChange(
                typeof value === "number" ? value : Number(value) || 3
              )
            }
          />
        </Tooltip>

        <Group gap="xs" wrap="nowrap">
          <Text fz="sm" c="dimmed">
            Heatmap
          </Text>
          <Group gap={4} wrap="nowrap">
            {COLORS.map((item) => (
              <Tooltip key={item.value} label={item.label} withArrow>
                <UnstyledButton
                  className={classes.swatch}
                  aria-label={`${item.label} heatmap`}
                  aria-pressed={props.color === item.value}
                  data-active={props.color === item.value || undefined}
                  style={{ backgroundColor: item.swatch }}
                  onClick={() => props.onColorChange(item.value)}
                />
              </Tooltip>
            ))}
          </Group>
          <Slider
            w={110}
            size="sm"
            aria-label="Heatmap opacity"
            label={(value) => `${value}%`}
            value={props.opacity}
            onChange={props.onOpacityChange}
          />
        </Group>
      </Group>
    </Group>
  );
}
