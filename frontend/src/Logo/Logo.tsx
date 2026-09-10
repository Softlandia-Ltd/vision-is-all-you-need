import { Image, useComputedColorScheme } from "@mantine/core";
import type { StyleProp } from "@mantine/core";

export interface LogoProps {
  h?: StyleProp<string | number>;
}

export function Logo({ h = 26 }: LogoProps) {
  const colorScheme = useComputedColorScheme("light");

  return (
    <Image
      h={h}
      w="auto"
      fit="contain"
      alt="Softlandia"
      src={
        colorScheme === "dark"
          ? "/softlandia-logo-white.svg"
          : "/softlandia-logo-black.svg"
      }
    />
  );
}
