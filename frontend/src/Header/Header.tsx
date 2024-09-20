import {
  Title,
  Text,
  Group,
  Image,
  useMantineColorScheme,
  ActionIcon,
  Tooltip,
} from "@mantine/core";
import * as classes from "./Header.css";
import { ColorSchemeToggle } from "../ColorSchemeToggle/ColorSchemeToggle";
import { IconBrandGithub } from "@tabler/icons-react";
import { IconArticle } from "@tabler/icons-react";

export default function Header() {
  const { colorScheme } = useMantineColorScheme();
  return (
    <header className={classes.header}>
      <Group w="100%" gap="lg">
        <Title className={classes.title} ta="center">
          <Text
            inherit
            variant="gradient"
            component="span"
            gradient={{ from: "pink", to: "yellow" }}
          >
            Vision is All You Need
          </Text>
        </Title>
        <Title ta="center">
          <Text
            fz="lg"
            c="dimmed"
            variant="gradient"
            gradient={{ from: "pink", to: "yellow" }}
          >
            by
          </Text>
        </Title>
        <a href="https://softlandia.fi">
          {colorScheme === "dark" ? (
            <Image
              h="30"
              w="auto"
              fit="contain"
              src={"/softlandia_logo_h_white_1.png"}
            />
          ) : (
            <Image
              h="30"
              w="auto"
              fit="contain"
              src={"/Softlandia_logo_Hor_color_small.png"}
            />
          )}
        </a>
        <Group ml={"auto"}>
          <Tooltip label="Background blog post">
            <ActionIcon
              size={"lg"}
              component="a"
              variant="default"
              aria-label="Background blog post"
              href="https://softlandia-ltd.github.io/vision-is-all-you-need/"
            >
              <IconArticle />
            </ActionIcon>
          </Tooltip>
          <Tooltip label="GitHub repository">
            <ActionIcon
              size={"lg"}
              component="a"
              variant="default"
              aria-label="GitHub repository"
              href="https://github.com/Softlandia-Ltd/vision-is-all-you-need"
            >
              <IconBrandGithub />
            </ActionIcon>
          </Tooltip>
          <ColorSchemeToggle />
        </Group>
      </Group>
    </header>
  );
}
