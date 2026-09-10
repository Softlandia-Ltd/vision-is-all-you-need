import { forwardRef } from "react";
import cx from "clsx";
import { Text, Group, Stack, Tooltip, ActionIcon, rem } from "@mantine/core";
import { Dropzone, MIME_TYPES } from "@mantine/dropzone";
import {
  IconCloudUpload,
  IconX,
  IconDownload,
  IconInfoCircle,
} from "@tabler/icons-react";
import * as classes from "./Dropzone.css";
import { INDEXING_NOTE } from "./notes";

export interface DropzoneBoxProps {
  uploading: boolean;
  onSubmit: (files: File[]) => void;
  openRef?: React.RefObject<() => void>;
  size?: "sm" | "lg";
}

export const DropzoneBox = forwardRef<HTMLDivElement, DropzoneBoxProps>(
  function DropzoneBox(props, ref) {
    const size = props.size ?? "sm";
    const iconSize = size === "lg" ? 40 : 32;

    return (
      <Dropzone
        ref={ref}
        disabled={props.uploading}
        openRef={props.openRef}
        onDrop={(files) => props.onSubmit(files)}
        className={cx(classes.dropzone, size === "lg" && classes.dropzoneLg)}
        radius="md"
        accept={[MIME_TYPES.pdf]}
        maxSize={30 * 1024 ** 2}
      >
        <Group justify="center" gap="sm" style={{ pointerEvents: "none" }}>
          <Dropzone.Accept>
            <IconDownload
              style={{ width: rem(iconSize), height: rem(iconSize) }}
              stroke={1.5}
            />
          </Dropzone.Accept>
          <Dropzone.Reject>
            <IconX
              style={{ width: rem(iconSize), height: rem(iconSize) }}
              color="var(--mantine-color-red-6)"
              stroke={1.5}
            />
          </Dropzone.Reject>
          <Dropzone.Idle>
            <IconCloudUpload
              style={{ width: rem(iconSize), height: rem(iconSize) }}
              stroke={1.5}
            />
          </Dropzone.Idle>
          <Stack gap={2}>
            <Text fw={600} fz={size === "lg" ? "md" : "sm"}>
              <Dropzone.Accept>Drop the files here</Dropzone.Accept>
              <Dropzone.Reject>PDF files under 30 MB only</Dropzone.Reject>
              <Dropzone.Idle>Drop PDFs here or click to browse</Dropzone.Idle>
            </Text>
            <Text fz="xs" c="dimmed">
              .pdf · max 30 MB
            </Text>
          </Stack>
        </Group>
      </Dropzone>
    );
  }
);

export function IndexingNote() {
  return (
    <Tooltip label={INDEXING_NOTE} multiline w={280} withArrow>
      <ActionIcon
        variant="subtle"
        color="gray"
        size="sm"
        aria-label="How indexing works"
      >
        <IconInfoCircle stroke={1.5} />
      </ActionIcon>
    </Tooltip>
  );
}
