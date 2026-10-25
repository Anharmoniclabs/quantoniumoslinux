import { Feather } from "@expo/vector-icons";
import { BlurView } from "expo-blur";
import * as Haptics from "expo-haptics";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import React, { useCallback, useRef, useState } from "react";
import {
  FlatList,
  Image,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
  useColorScheme,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { ChatBubble } from "@/components/ChatBubble";
import { TypingIndicator } from "@/components/TypingIndicator";
import { useChat } from "@/contexts/ChatContext";
import { useColors } from "@/hooks/useColors";

const LOGO = require("@/assets/images/quai-logo.png");

export default function ChatScreen() {
  const colors = useColors();
  const colorScheme = useColorScheme();
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { messages, isTyping, sendMessage, availableModels } = useChat();
  const [input, setInput] = useState("");
  const [inputFocused, setInputFocused] = useState(false);
  const listRef = useRef<FlatList>(null);
  const isWeb = Platform.OS === "web";
  const isIOS = Platform.OS === "ios";
  const isDark = colorScheme === "dark";

  const connected = availableModels.some((m) => m.backend !== "demo");

  const webTopPad = isWeb ? 67 : 0;
  const webBottomPad = isWeb ? 34 : 0;
  const headerHeight = (isWeb ? webTopPad : insets.top) + 60;

  const gradientColors = isDark
    ? ([colors.gradientStart, colors.gradientMid, colors.gradientEnd] as const)
    : (["#a8bfe0", "#c0cfe8", "#d8e4f6", "#e8eefb", "#f2f5ff"] as const);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text) return;
    setInput("");
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    await sendMessage(text);
  }, [input, sendMessage]);

  const renderItem = useCallback(
    ({ item }: { item: (typeof messages)[0] }) => <ChatBubble message={item} />,
    []
  );

  /* Glass tab colours */
  const tabBg = isDark ? "rgba(10,16,30,0.50)" : "rgba(255,255,255,0.35)";
  const tabActiveBg = isDark ? "rgba(0,212,255,0.10)" : "rgba(255,255,255,0.70)";
  const tabBorder = isDark ? "rgba(60,100,160,0.20)" : "rgba(200,220,255,0.50)";
  const tabActiveBorder = isDark ? "rgba(0,212,255,0.30)" : "rgba(100,150,230,0.40)";

  /* Header glass */
  const headerBg = isDark ? "rgba(6,8,16,0.65)" : "rgba(220,230,250,0.55)";
  const headerBorder = isDark ? "rgba(60,100,160,0.15)" : "rgba(200,220,255,0.40)";

  /* Input glass */
  const inputBg = isDark ? "rgba(10,16,30,0.60)" : "rgba(255,255,255,0.55)";
  const inputBorder = inputFocused
    ? (isDark ? "rgba(0,212,255,0.40)" : "rgba(0,160,200,0.35)")
    : (isDark ? "rgba(60,100,160,0.20)" : "rgba(200,220,255,0.45)");
  const inputShadow = isDark ? "rgba(0,180,255,0.12)" : "rgba(0,140,200,0.10)";

  /* Send button */
  const sendBg = input.trim() && !isTyping
    ? (isDark ? colors.glow : "#4a7eb5")
    : (isDark ? "rgba(30,50,80,0.40)" : "rgba(180,200,235,0.50)");

  return (
    <LinearGradient
      colors={gradientColors}
      style={styles.root}
      start={{ x: 0.2, y: 0 }}
      end={{ x: 0.8, y: 1 }}
    >
      {/* Header */}
      <View style={[styles.headerWrap, { height: headerHeight, borderBottomColor: headerBorder }]}>
        {isIOS ? (
          <BlurView intensity={40} tint={isDark ? "dark" : "light"} style={StyleSheet.absoluteFill} />
        ) : (
          <View style={[StyleSheet.absoluteFill, { backgroundColor: headerBg }]} />
        )}
        <View style={[styles.header, { paddingTop: isWeb ? webTopPad : insets.top }]}>

          {/* Tab switcher */}
          <View style={styles.tabSwitcher}>
            <View style={[styles.tabPill, { borderColor: tabBorder, backgroundColor: tabBg }]}>
              <View style={[styles.tabBtn, styles.tabBtnActive, { backgroundColor: tabActiveBg, borderColor: tabActiveBorder }]}>
                <Feather name="message-circle" size={13} color={isDark ? colors.glow : "#2a5ab0"} />
                <Text style={[styles.tabLabel, styles.tabLabelActive, { color: isDark ? colors.glow : "#2a5ab0" }]}>
                  Chat
                </Text>
              </View>
              <Pressable
                style={[styles.tabBtn, { backgroundColor: "transparent", borderColor: "transparent" }]}
                onPress={() => {
                  Haptics.selectionAsync().catch(() => {});
                  router.navigate("/settings");
                }}
                hitSlop={4}
              >
                <Feather name="settings" size={13} color={isDark ? colors.mutedForeground : "#7090c0"} />
                <Text style={[styles.tabLabel, { color: isDark ? colors.mutedForeground : "#7090c0" }]}>
                  Settings
                </Text>
              </Pressable>
            </View>
          </View>

          {/* Center logo + name */}
          <View style={styles.headerCenter}>
            <View style={[styles.logoInline, { shadowColor: isDark ? colors.glow : "rgba(0,140,200,0.25)" }]}>
              <Image source={LOGO} style={styles.logoImg} resizeMode="contain" />
            </View>
            <Text style={[styles.headerTitle, { color: isDark ? "#e0eeff" : "#2a4a80" }]}>
              QU.Ai
            </Text>
            {/* Connection indicator dot */}
            <View style={[
              styles.connDot,
              { backgroundColor: connected ? colors.safetyGreen : colors.safetyRed },
              connected && { shadowColor: colors.safetyGreen, shadowOpacity: 0.6, shadowRadius: 4 },
            ]} />
          </View>

          {/* Right action */}
          <Pressable style={styles.headerSideBtn} hitSlop={8}>
            <Feather name="more-horizontal" size={22} color={isDark ? colors.primary : "#3a5080"} />
          </Pressable>
        </View>
      </View>

      <KeyboardAvoidingView
        style={styles.flex}
        behavior={isIOS ? "padding" : "height"}
        keyboardVerticalOffset={0}
      >
        <FlatList
          ref={listRef}
          data={messages}
          renderItem={renderItem}
          keyExtractor={(item) => item.id}
          contentContainerStyle={[
            styles.listContent,
            { paddingTop: headerHeight + 8, paddingBottom: insets.bottom + 96 },
          ]}
          onContentSizeChange={() => listRef.current?.scrollToEnd({ animated: true })}
          onLayout={() => listRef.current?.scrollToEnd({ animated: false })}
          showsVerticalScrollIndicator={false}
          scrollIndicatorInsets={{ right: 1 }}
          ListEmptyComponent={
            <View style={styles.emptyState}>
              <View style={[styles.emptyLogoWrap, { shadowColor: isDark ? colors.glow : "rgba(0,140,200,0.20)" }]}>
                <Image source={LOGO} style={styles.emptyLogo} resizeMode="contain" />
              </View>
              <Text style={[styles.emptyTitle, { color: isDark ? "#e0eeff" : "#2a4a80" }]}>
                How can I help you?
              </Text>
              <Text style={[styles.emptyDesc, { color: isDark ? colors.mutedForeground : "#6080b0" }]}>
                Local AI inference · Private by default
              </Text>
            </View>
          }
          ListFooterComponent={isTyping ? <TypingIndicator /> : null}
        />

        {/* Input pill */}
        <View
          style={[
            styles.inputOuter,
            { paddingBottom: isWeb ? webBottomPad + 12 : insets.bottom + 12 },
          ]}
        >
          <View
            style={[
              styles.inputPill,
              {
                backgroundColor: inputBg,
                borderColor: inputBorder,
                shadowColor: inputShadow,
                shadowOpacity: inputFocused ? 0.25 : 0.10,
              },
            ]}
          >
            <Feather name="paperclip" size={18} color={isDark ? colors.mutedForeground : "#90a8c8"} />
            <TextInput
              style={[styles.input, { color: isDark ? colors.foreground : "#1a2050" }]}
              placeholder="Ask anything..."
              placeholderTextColor={isDark ? colors.mutedForeground : "#90a8c8"}
              value={input}
              onChangeText={setInput}
              multiline
              maxLength={4000}
              onFocus={() => setInputFocused(true)}
              onBlur={() => setInputFocused(false)}
              onSubmitEditing={isWeb ? handleSend : undefined}
              blurOnSubmit={false}
            />
            <Pressable
              onPress={handleSend}
              disabled={!input.trim() || isTyping}
              style={({ pressed }) => [
                styles.sendBtn,
                {
                  backgroundColor: sendBg,
                  opacity: pressed ? 0.75 : 1,
                },
              ]}
            >
              <Feather
                name="send"
                size={16}
                color={input.trim() && !isTyping ? "#ffffff" : (isDark ? "#3d5470" : "#90a8c8")}
              />
            </Pressable>
          </View>
        </View>
      </KeyboardAvoidingView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  flex: { flex: 1 },

  headerWrap: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    zIndex: 10,
    overflow: "hidden",
    borderBottomWidth: 1,
  },
  header: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 12,
    paddingBottom: 10,
  },

  tabSwitcher: {
    flexDirection: "row",
    alignItems: "center",
  },
  tabPill: {
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1,
    borderRadius: 20,
    padding: 3,
    gap: 2,
  },
  tabBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 16,
    borderWidth: 1,
  },
  tabBtnActive: {},
  tabLabel: {
    fontSize: 12,
    fontFamily: "Inter_400Regular",
    fontWeight: "500",
  },
  tabLabelActive: {
    fontWeight: "700",
    fontFamily: "Inter_700Bold",
  },

  headerCenter: {
    flexDirection: "row",
    alignItems: "center",
    gap: 7,
  },
  logoInline: {
    width: 30,
    height: 30,
    borderRadius: 8,
    overflow: "hidden",
    backgroundColor: "#060810",
    shadowOpacity: 0.5,
    shadowRadius: 8,
    shadowOffset: { width: 0, height: 0 },
    elevation: 4,
  },
  logoImg: { width: 30, height: 30 },
  headerTitle: {
    fontSize: 20,
    fontWeight: "700",
    fontFamily: "Inter_700Bold",
    letterSpacing: 0.3,
  },
  connDot: {
    width: 7,
    height: 7,
    borderRadius: 3.5,
    marginLeft: 2,
    shadowOffset: { width: 0, height: 0 },
    elevation: 2,
  },
  headerSideBtn: {
    width: 36,
    height: 36,
    alignItems: "center",
    justifyContent: "center",
  },

  listContent: { paddingHorizontal: 0 },

  emptyState: {
    alignItems: "center",
    paddingTop: 60,
    paddingHorizontal: 24,
    gap: 12,
  },
  emptyLogoWrap: {
    width: 140,
    height: 140,
    borderRadius: 28,
    overflow: "hidden",
    backgroundColor: "#060810",
    marginBottom: 10,
    shadowOpacity: 0.6,
    shadowRadius: 20,
    shadowOffset: { width: 0, height: 0 },
    elevation: 8,
  },
  emptyLogo: { width: 140, height: 140 },
  emptyTitle: {
    fontSize: 24,
    fontWeight: "700",
    fontFamily: "Inter_700Bold",
    letterSpacing: -0.3,
  },
  emptyDesc: {
    fontSize: 14,
    fontFamily: "Inter_400Regular",
    textAlign: "center",
    lineHeight: 20,
  },

  inputOuter: {
    paddingHorizontal: 14,
    paddingTop: 8,
  },
  inputPill: {
    flexDirection: "row",
    alignItems: "center",
    borderWidth: 1,
    borderRadius: 30,
    paddingLeft: 16,
    paddingRight: 6,
    paddingVertical: 8,
    gap: 10,
    shadowRadius: 16,
    shadowOffset: { width: 0, height: 4 },
    elevation: 4,
  },
  input: {
    flex: 1,
    fontSize: 15,
    fontFamily: "Inter_400Regular",
    lineHeight: 22,
    maxHeight: 120,
    paddingVertical: 0,
  },
  sendBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
});
