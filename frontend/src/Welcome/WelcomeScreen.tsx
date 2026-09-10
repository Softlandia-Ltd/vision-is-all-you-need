import { useRef } from "react";
import { Alert, Group, Loader, Stack, Text, Title } from "@mantine/core";
import { IconAlertTriangle } from "@tabler/icons-react";
import { DropzoneBox, IndexingNote } from "../Dropzone/Dropzone";
import { Question } from "../Question/Question";
import { Logo } from "../Logo/Logo";
import * as classes from "./WelcomeScreen.css";

export interface WelcomeScreenProps {
  uploading: boolean;
  uploadStatus: string;
  error: string | null;
  onSubmitFiles: (files: File[]) => void;
}

export function WelcomeScreen(props: WelcomeScreenProps) {
  const openRef = useRef<() => void>(null);

  return (
    <div className={classes.screen}>
      <Stack className={classes.content} gap="lg" align="stretch">
        <Stack gap={4} align="center">
          <Title order={1} ta="center" className={classes.heading}>
            Vision RAG
          </Title>
          <Text ta="center" className={classes.tagline}>
            Vision is all you need
          </Text>
          <Group gap={8} justify="center" pt={6}>
            <Text c="dimmed" fz="sm">
              by
            </Text>
            <a href="https://softlandia.fi" aria-label="Softlandia">
              <Logo h={22} />
            </a>
          </Group>
        </Stack>

        <Question
          onSubmit={() => undefined}
          loading={false}
          disabled
          size="lg"
          placeholder="Upload a PDF to start asking questions"
          onDisabledClick={() => openRef.current?.()}
        />

        {props.error && (
          <Alert
            color="red"
            variant="light"
            icon={<IconAlertTriangle size={18} />}
            title="Upload failed"
          >
            <Text fz="xs" style={{ overflowWrap: "anywhere" }}>
              {props.error}
            </Text>
          </Alert>
        )}
        {props.uploading ? (
          <Group justify="center" gap="sm" className={classes.status}>
            <Loader size="sm" />
            <Text fz="md" c="dimmed">
              {props.uploadStatus || "Indexing…"}
            </Text>
          </Group>
        ) : (
          <Stack gap="xs">
            <DropzoneBox
              size="lg"
              uploading={props.uploading}
              openRef={openRef}
              onSubmit={props.onSubmitFiles}
            />
            <Group gap={6} justify="center">
              <Text fz="xs" c="dimmed">
                Indexes are temporary and live only while the demo container runs
              </Text>
              <IndexingNote />
            </Group>
          </Stack>
        )}
      </Stack>
    </div>
  );
}
