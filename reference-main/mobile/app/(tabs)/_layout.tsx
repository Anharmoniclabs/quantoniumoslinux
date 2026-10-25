import { Tabs } from "expo-router";
import React from "react";

/**
 * Tab bar is fully hidden — navigation is handled by the custom glass
 * pill switchers inside index.tsx and settings.tsx.
 */
export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          display: "none",
          position: "absolute",
          height: 0,
          width: 0,
          opacity: 0,
          overflow: "hidden",
        },
        tabBarShowLabel: false,
        tabBarItemStyle: { height: 0 },
      }}
    >
      <Tabs.Screen name="index" options={{ title: "Chat" }} />
      <Tabs.Screen name="settings" options={{ title: "Settings" }} />
    </Tabs>
  );
}
