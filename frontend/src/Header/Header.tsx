import { Text, Group, ActionIcon, Tooltip } from "@mantine/core";
import * as classes from "./Header.css";
import { ColorSchemeToggle } from "../ColorSchemeToggle/ColorSchemeToggle";
import { Logo } from "../Logo/Logo";
import { IconBrandGithub, IconArticle } from "@tabler/icons-react";

export default function Header() {
  return (
    <header className={classes.header}>
      <Group gap="sm" wrap="nowrap" className={classes.brand}>
        <a href="https://softlandia.fi" aria-label="Softlandia">
          <Logo h={{ base: 20, sm: 26 }} />
        </a>
        <div className={classes.separator} />
        <Text className={classes.title} lineClamp={1}>
          Vision is All You Need
        </Text>
      </Group>
      <Group gap="xs" wrap="nowrap">
        <Tooltip label="Background blog post">
          <ActionIcon
            size="lg"
            component="a"
            variant="subtle"
            color="gray"
            aria-label="Background blog post"
            href="https://softlandia.fi/en/blog/building-a-rag-tired-of-chunking-maybe-vision-is-all-you-need"
          >
            <IconArticle stroke={1.5} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="GitHub repository">
          <ActionIcon
            size="lg"
            component="a"
            variant="subtle"
            color="gray"
            aria-label="GitHub repository"
            href="https://github.com/Softlandia-Ltd/vision-is-all-you-need"
          >
            <IconBrandGithub stroke={1.5} />
          </ActionIcon>
        </Tooltip>
        <ColorSchemeToggle />
      </Group>
    </header>
  );
}
