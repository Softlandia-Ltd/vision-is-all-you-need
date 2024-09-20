import { useRef } from "react";
import { Text, Group, rem, useMantineTheme } from "@mantine/core";
import { Dropzone, MIME_TYPES } from "@mantine/dropzone";
import { IconCloudUpload, IconX, IconDownload } from "@tabler/icons-react";
import * as classes from "./Dropzone.css";

export interface DropzoneBoxProps {
  uploading: boolean;
  onSubmit: (files: File[]) => void;
}

export function DropzoneBox(props: DropzoneBoxProps) {
  const theme = useMantineTheme();
  const openRef = useRef<() => void>(null);

  return (
    <div className={classes.wrapper}>
      <Dropzone
        disabled={props.uploading}
        openRef={openRef}
        onDrop={(files) => props.onSubmit(files)}
        className={classes.dropzone}
        radius="md"
        accept={[MIME_TYPES.pdf]}
        maxSize={30 * 1024 ** 2}
      >
        <div style={{ pointerEvents: "none" }}>
          <Group justify="center">
            <Dropzone.Accept>
              <IconDownload
                style={{ width: rem(50), height: rem(50) }}
                color={theme.colors.blue[6]}
                stroke={1.5}
              />
            </Dropzone.Accept>
            <Dropzone.Reject>
              <IconX
                style={{ width: rem(50), height: rem(50) }}
                color={theme.colors.red[6]}
                stroke={1.5}
              />
            </Dropzone.Reject>
            <Dropzone.Idle>
              <IconCloudUpload
                style={{ width: rem(50), height: rem(50) }}
                stroke={1.5}
              />
            </Dropzone.Idle>
          </Group>

          <Text ta="center" fw={700} fz="lg">
            <Dropzone.Accept>Drop files here</Dropzone.Accept>
            <Dropzone.Reject>Pdf file less than 30mb</Dropzone.Reject>
            <Dropzone.Idle>Upload PDFs</Dropzone.Idle>
          </Text>
          <Text ta="center" fz="sm" c="dimmed">
            Only <i>.pdf</i> files that are less than 30mb in size are accepted.
            There's no persistent storage, the index is built on the fly and
            will be lost when the underlying container dies. <br /> If the
            upload takes long, it's likely that the container has been
            terminated and you need to wait for a little while for the new
            container to be up and running.
          </Text>
        </div>
      </Dropzone>
    </div>
  );
}
