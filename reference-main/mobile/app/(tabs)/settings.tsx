import { Feather } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import { LinearGradient } from "expo-linear-gradient";
import { useRouter } from "expo-router";
import React, { useEffect, useRef, useState } from "react";
import {
  Alert,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
  useColorScheme,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { useChat } from "@/contexts/ChatContext";
import { useColors } from "@/hooks/useColors";

export default function SettingsScreen() {
  const colors = useColors();
  const colorScheme = useColorScheme();
  const isDark = colorScheme === "dark";
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const {
    ollamaUrl,
    setOllamaUrl,
    selectedModel,
    setSelectedModel,
    availableModels,
    safeMode,
    setSafeMode,
    safetyOk,
    clearHistory,
    refreshModels,
  } = useChat();

  const [urlDraft, setUrlDraft] = useState(ollamaUrl);
  const [saving, setSaving] = useState(false);
  const isWeb = Platform.OS === "web";
  const userEdited = useRef(false);

  // Sync urlDraft when ollamaUrl changes (e.g. AsyncStorage restore)
  // but not after the user has manually edited the field
  useEffect(() => {
    if (!userEdited.current) setUrlDraft(ollamaUrl);
  }, [ollamaUrl]);

  const saveUrl = async () => {
    const url = urlDraft.trim();
    setSaving(true);
    try {
      await setOllamaUrl(url);
      // Pass URL directly to avoid stale closure
      await refreshModels(url);
      userEdited.current = false;
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
    } catch {} finally {
      setSaving(false);
    }
  };

  const handleClear = () => {
    if (Platform.OS === "web") {
      if (window.confirm("Clear all conversation history?")) clearHistory();
      return;
    }
    Alert.alert("Clear History", "Delete all conversation history?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Clear",
        style: "destructive",
        onPress: () => {
          clearHistory();
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning).catch(() => {});
        },
      },
    ]);
  };

  const topPad = isWeb ? 80 : insets.top + 12;
  const bottomPad = isWeb ? 50 : insets.bottom + 24;

  /* Glass tab colours */
  const tabBg = isDark ? "rgba(10,16,30,0.50)" : "rgba(255,255,255,0.35)";
  const tabActiveBg = isDark ? "rgba(0,212,255,0.10)" : "rgba(255,255,255,0.70)";
  const tabBorder = isDark ? "rgba(60,100,160,0.20)" : "rgba(200,220,255,0.50)";
  const tabActiveBorder = isDark ? "rgba(0,212,255,0.30)" : "rgba(100,150,230,0.40)";

  /* Glass card */
  const cardBg = isDark ? "rgba(10,16,30,0.50)" : "rgba(255,255,255,0.45)";
  const cardBorder = isDark ? "rgba(60,100,160,0.20)" : "rgba(200,220,255,0.45)";
  const cardShadow = isDark ? "rgba(0,180,255,0.06)" : "rgba(0,140,200,0.08)";

  const gradientColors = isDark
    ? ([colors.gradientStart, colors.gradientMid, colors.gradientEnd] as const)
    : (["#a8bfe0", "#c0cfe8", "#d8e4f6", "#e8eefb", "#f2f5ff"] as const);

  /* Status glow */
  const statusColor = safetyOk ? colors.safetyGreen : colors.safetyRed;

  return (
    <LinearGradient
      colors={gradientColors}
      style={styles.root}
      start={{ x: 0.2, y: 0 }}
      end={{ x: 0.8, y: 1 }}
    >
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={[styles.content, { paddingTop: topPad, paddingBottom: bottomPad }]}
        showsVerticalScrollIndicator={false}
      >
        {/* Tab switcher */}
        <View style={[styles.tabSwitcher, { marginBottom: 20 }]}>
          <View style={[styles.tabPill, { borderColor: tabBorder, backgroundColor: tabBg }]}>
            <Pressable
              style={[styles.tabBtn, { backgroundColor: "transparent", borderColor: "transparent" }]}
              onPress={() => {
                Haptics.selectionAsync().catch(() => {});
                router.navigate("/");
              }}
              hitSlop={4}
            >
              <Feather name="message-circle" size={13} color={isDark ? colors.mutedForeground : "#7090c0"} />
              <Text style={[styles.tabLabel, { color: isDark ? colors.mutedForeground : "#7090c0" }]}>
                Chat
              </Text>
            </Pressable>
            <View style={[styles.tabBtn, styles.tabBtnActive, { backgroundColor: tabActiveBg, borderColor: tabActiveBorder }]}>
              <Feather name="settings" size={13} color={isDark ? colors.glow : "#2a5ab0"} />
              <Text style={[styles.tabLabel, styles.tabLabelActive, { color: isDark ? colors.glow : "#2a5ab0" }]}>
                Settings
              </Text>
            </View>
          </View>
        </View>

        <Text style={[styles.pageTitle, { color: isDark ? "#e0eeff" : colors.foreground }]}>Settings</Text>

        {/* Connection */}
        <Text style={[styles.sectionLabel, { color: colors.mutedForeground }]}>CONNECTION</Text>
        <View style={[styles.card, { backgroundColor: cardBg, borderColor: cardBorder, shadowColor: cardShadow }]}>
          <View style={styles.cardRow}>
            <View style={[styles.iconBox, { backgroundColor: colors.primary + "18" }]}>
              <Feather name="server" size={15} color={colors.primary} />
            </View>
            <View style={styles.rowBody}>
              <Text style={[styles.rowLabel, { color: colors.foreground }]}>Model Server URL</Text>
              <View style={styles.urlRow}>
                <TextInput
                  style={[styles.urlInput, {
                    color: colors.foreground,
                    backgroundColor: isDark ? "rgba(15,25,45,0.50)" : "rgba(255,255,255,0.40)",
                    borderColor: isDark ? "rgba(60,100,160,0.25)" : "rgba(200,220,255,0.50)",
                  }]}
                  value={urlDraft}
                  onChangeText={(t) => { userEdited.current = true; setUrlDraft(t); }}
                  autoCapitalize="none"
                  autoCorrect={false}
                  placeholder="http://localhost:8000"
                  placeholderTextColor={colors.mutedForeground}
                />
                <Pressable
                  onPress={saveUrl}
                  style={({ pressed }) => [
                    styles.saveBtn,
                    { backgroundColor: isDark ? colors.glow : colors.primary, opacity: pressed ? 0.8 : 1 },
                  ]}
                >
                  <Text style={[styles.saveBtnText, { color: isDark ? "#060810" : colors.primaryForeground }]}>
                    {saving ? "…" : "Save"}
                  </Text>
                </Pressable>
              </View>
            </View>
          </View>

          <View style={[styles.divider, { backgroundColor: cardBorder }]} />

          <View style={styles.cardRow}>
            <View style={[styles.iconBox, { backgroundColor: statusColor + "18" }]}>
              <Feather name="activity" size={15} color={statusColor} />
            </View>
            <View style={styles.rowBody}>
              <Text style={[styles.rowLabel, { color: colors.foreground }]}>Status</Text>
            </View>
            <View style={[styles.statusPill, {
              backgroundColor: statusColor + "18",
              shadowColor: safetyOk ? colors.safetyGreen : "transparent",
              shadowOpacity: safetyOk ? 0.4 : 0,
              shadowRadius: safetyOk ? 6 : 0,
            }]}>
              <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
              <Text style={[styles.statusText, { color: statusColor }]}>
                {safetyOk ? "Connected" : "Offline"}
              </Text>
            </View>
          </View>
        </View>

        {/* Model */}
        <Text style={[styles.sectionLabel, { color: colors.mutedForeground }]}>MODEL</Text>
        <View style={[styles.card, { backgroundColor: cardBg, borderColor: cardBorder, shadowColor: cardShadow }]}>
          <View style={styles.modelsInner}>
            {availableModels.length === 0 ? (
              <Text style={[styles.noModels, { color: colors.mutedForeground }]}>
                No models found. Start the server, then refresh.
              </Text>
            ) : (
              availableModels.map((m) => {
                const active = selectedModel === m.id;
                return (
                  <Pressable
                    key={m.id}
                    onPress={() => { setSelectedModel(m.id); Haptics.selectionAsync().catch(() => {}); }}
                    style={({ pressed }) => [
                      styles.modelItem,
                      {
                        backgroundColor: active
                          ? (isDark ? "rgba(0,212,255,0.08)" : colors.primary + "18")
                          : pressed
                          ? (isDark ? "rgba(20,36,60,0.40)" : colors.muted)
                          : "transparent",
                        borderColor: active
                          ? (isDark ? "rgba(0,212,255,0.25)" : colors.primary + "60")
                          : cardBorder,
                      },
                    ]}
                  >
                    <Feather name="cpu" size={14} color={active ? (isDark ? colors.glow : colors.primary) : colors.mutedForeground} />
                    <Text
                      style={[styles.modelLabel, { color: active ? (isDark ? colors.glow : colors.primary) : colors.foreground }]}
                      numberOfLines={1}
                    >
                      {m.label}
                    </Text>
                    {m.backend === "rftmw" && (
                      <View style={[styles.backendBadge, {
                        backgroundColor: isDark ? "rgba(0,212,255,0.10)" : colors.primary + "22",
                        borderColor: isDark ? "rgba(0,212,255,0.25)" : colors.primary + "44",
                      }]}>
                        <Text style={[styles.backendBadgeText, { color: isDark ? colors.glow : colors.primary }]}>RFTMW</Text>
                      </View>
                    )}
                    {active && <Feather name="check-circle" size={15} color={isDark ? colors.glow : colors.primary} />}
                  </Pressable>
                );
              })
            )}
          </View>
          <Pressable
            onPress={async () => { await refreshModels(); Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {}); }}
            style={[styles.refreshBtn, { borderTopColor: cardBorder }]}
          >
            <Feather name="refresh-cw" size={13} color={colors.mutedForeground} />
            <Text style={[styles.refreshText, { color: colors.mutedForeground }]}>Refresh Models</Text>
          </Pressable>
        </View>

        {/* Safety */}
        <Text style={[styles.sectionLabel, { color: colors.mutedForeground }]}>SAFETY</Text>
        <View style={[styles.card, { backgroundColor: cardBg, borderColor: cardBorder, shadowColor: cardShadow }]}>
          <View style={styles.cardRow}>
            <View style={[styles.iconBox, { backgroundColor: colors.accent + "18" }]}>
              <Feather name="shield" size={15} color={colors.accent} />
            </View>
            <View style={styles.rowBody}>
              <Text style={[styles.rowLabel, { color: colors.foreground }]}>Safe Mode</Text>
              <Text style={[styles.rowSub, { color: colors.mutedForeground }]}>
                Blocks system commands
              </Text>
            </View>
            <Switch
              value={safeMode}
              onValueChange={(v) => { setSafeMode(v); Haptics.selectionAsync().catch(() => {}); }}
              trackColor={{ false: isDark ? "rgba(30,50,80,0.50)" : colors.border, true: isDark ? colors.glow : colors.primary }}
              thumbColor="#fff"
            />
          </View>
        </View>

        {/* Data */}
        <Text style={[styles.sectionLabel, { color: colors.mutedForeground }]}>DATA</Text>
        <Pressable
          onPress={handleClear}
          style={({ pressed }) => [
            styles.dangerBtn,
            {
              backgroundColor: pressed ? colors.destructive + "20" : colors.destructive + "10",
              borderColor: colors.destructive + "40",
            },
          ]}
        >
          <Feather name="trash-2" size={16} color={colors.destructive} />
          <Text style={[styles.dangerText, { color: colors.destructive }]}>
            Clear Conversation History
          </Text>
        </Pressable>

        {/* About */}
        <View style={[styles.aboutWrap, { borderColor: cardBorder }]}>
          <Text style={[styles.aboutName, { color: isDark ? colors.glow : colors.primary }]}>QU.Ai Mobile</Text>
          <Text style={[styles.aboutDesc, { color: colors.mutedForeground }]}>
            Local-first AI powered by RFTMW · Privacy-first
          </Text>
          <Text style={[styles.aboutVer, { color: colors.mutedForeground }]}>v1.0.0</Text>
        </View>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  scroll: { flex: 1 },
  content: { paddingHorizontal: 18 },
  pageTitle: {
    fontSize: 30,
    fontWeight: "700",
    fontFamily: "Inter_700Bold",
    letterSpacing: -0.5,
    marginBottom: 24,
  },
  sectionLabel: {
    fontSize: 11,
    fontWeight: "700",
    fontFamily: "Inter_700Bold",
    letterSpacing: 1.2,
    marginBottom: 8,
    paddingLeft: 2,
  },
  card: {
    borderRadius: 16,
    borderWidth: 1,
    marginBottom: 28,
    overflow: "hidden",
    shadowOpacity: 0.12,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: 4 },
    elevation: 3,
  },
  cardRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    padding: 14,
  },
  iconBox: {
    width: 34,
    height: 34,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  rowBody: { flex: 1 },
  rowLabel: {
    fontSize: 15,
    fontWeight: "500",
    fontFamily: "Inter_500Medium",
  },
  rowSub: {
    fontSize: 12,
    fontFamily: "Inter_400Regular",
    marginTop: 1,
  },
  divider: { height: StyleSheet.hairlineWidth, marginHorizontal: 14 },
  urlRow: { flexDirection: "row", gap: 8, marginTop: 8, alignItems: "center" },
  urlInput: {
    flex: 1,
    fontSize: 13,
    fontFamily: "Inter_400Regular",
    borderWidth: 1,
    borderRadius: 10,
    paddingHorizontal: 11,
    height: 36,
  },
  saveBtn: {
    paddingHorizontal: 16,
    minWidth: 60,
    height: 36,
    borderRadius: 10,
    justifyContent: "center",
    alignItems: "center",
  },
  saveBtnText: { fontSize: 13, fontWeight: "600", fontFamily: "Inter_600SemiBold" },
  statusPill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
  },
  statusDot: { width: 6, height: 6, borderRadius: 3 },
  statusText: { fontSize: 12, fontWeight: "600", fontFamily: "Inter_600SemiBold" },
  modelsInner: { padding: 10, gap: 4 },
  modelItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingHorizontal: 12,
    paddingVertical: 11,
    borderRadius: 12,
    borderWidth: 1,
  },
  modelLabel: {
    flex: 1,
    fontSize: 14,
    fontFamily: "Inter_400Regular",
  },
  noModels: {
    fontSize: 13,
    fontFamily: "Inter_400Regular",
    textAlign: "center",
    paddingVertical: 10,
  },
  refreshBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    paddingVertical: 12,
    borderTopWidth: StyleSheet.hairlineWidth,
  },
  refreshText: { fontSize: 13, fontFamily: "Inter_400Regular" },
  dangerBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    paddingVertical: 15,
    borderRadius: 16,
    borderWidth: 1,
    marginBottom: 28,
  },
  dangerText: {
    fontSize: 15,
    fontWeight: "600",
    fontFamily: "Inter_600SemiBold",
  },
  aboutWrap: {
    alignItems: "center",
    paddingVertical: 20,
    borderTopWidth: StyleSheet.hairlineWidth,
    gap: 4,
  },
  aboutName: { fontSize: 14, fontWeight: "600", fontFamily: "Inter_600SemiBold" },
  aboutDesc: { fontSize: 12, fontFamily: "Inter_400Regular", textAlign: "center", lineHeight: 17 },
  aboutVer: { fontSize: 11, fontFamily: "Inter_400Regular", marginTop: 4 },

  tabSwitcher: { flexDirection: "row" },
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
  tabLabel: { fontSize: 12, fontFamily: "Inter_400Regular", fontWeight: "500" },
  tabLabelActive: { fontWeight: "700", fontFamily: "Inter_700Bold" },
  backendBadge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 6,
    borderWidth: 1,
  },
  backendBadgeText: {
    fontSize: 9,
    fontWeight: "700",
    fontFamily: "Inter_700Bold",
    letterSpacing: 0.6,
  },
});
