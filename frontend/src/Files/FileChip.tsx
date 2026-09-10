import { Group, Text, Tooltip, ActionIcon } from "@mantine/core";
import { IconFileTypePdf, IconX } from "@tabler/icons-react";
import * as classes from "./FileChip.css";

export interface FileChipProps {
  name: string;
  onRemove?: () => void;
}

export function FileChip({ name, onRemove }: FileChipProps) {
  return (
    <Group gap={6} wrap="nowrap" className={classes.chip}>
      <IconFileTypePdf
        size={18}
        stroke={1.6}
        color="var(--mantine-color-red-6)"
      />
      <Tooltip label={name} withArrow openDelay={400}>
        <Text fz="sm" fw={500} truncate maw={200}>
          {name}
        </Text>
      </Tooltip>
      {onRemove && (
        <ActionIcon
          size={16}
          variant="subtle"
          color="gray"
          aria-label={`Remove ${name}`}
          onClick={onRemove}
        >
          <IconX size={12} stroke={2} />
        </ActionIcon>
      )}
    </Group>
  );
}
