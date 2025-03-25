import { Link } from "expo-router";
import React, { useState, useEffect } from "react";
import {
  Text,
  View,
  Button,
  StyleSheet,
  Switch,
  SafeAreaView,
  ActivityIndicator,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { useEvent } from "expo";
import { useVideoPlayer, VideoView } from "expo-video";
import * as FileSystem from "expo-file-system";
import * as MediaLibrary from "expo-media-library";

// Logger utility for consistent logging
const Logger = {
  info: (message: string, ...args: any[]) => {
    if (__DEV__) {
      console.log(`[INFO] ${message}`, ...args);
    }
  },
  warning: (message: string, ...args: any[]) => {
    if (__DEV__) {
      console.warn(`[WARNING] ${message}`, ...args);
    }
  },
  error: (message: string, error?: any) => {
    // Always log errors, even in production
    console.error(`[ERROR] ${message}`, error || "");
    // In a real app, you might want to report errors to a service
    // like Sentry, Firebase Crashlytics, etc.
  },
  debug: (message: string, ...args: any[]) => {
    if (__DEV__) {
      console.log(`[DEBUG] ${message}`, ...args);
    }
  },
};

interface ContentProps {
  pickVideo: () => Promise<void>;
  videoUri: string | null;
  watermarkDetected: boolean | null;
  processedVideoUri: string | null;
  useInpainting: boolean;
  setUseInpainting: (value: boolean) => void;
}

// Maximum number of local cached videos to keep
const MAX_LOCAL_VIDEOS = 2;

export default function Page() {
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [watermarkDetected, setWatermarkDetected] = useState<boolean | null>(
    null
  );
  const [processedVideoUri, setProcessedVideoUri] = useState<string | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useInpainting, setUseInpainting] = useState(false);

  // Clean up local storage on component mount
  useEffect(() => {
    cleanupLocalVideos();
  }, []);

  const cleanupLocalVideos = async () => {
    try {
      // Get the document directory contents
      const docDir = FileSystem.documentDirectory;
      const files = await FileSystem.readDirectoryAsync(docDir);

      // Filter out mp4 files (our processed videos)
      const videoFiles = files.filter((file) => file.endsWith(".mp4"));
      Logger.info(`Found ${videoFiles.length} cached videos in local storage`);

      // If we have more than MAX_LOCAL_VIDEOS, delete the oldest ones
      if (videoFiles.length > MAX_LOCAL_VIDEOS) {
        // We need to get file info to sort by creation time
        const fileInfos = await Promise.all(
          videoFiles.map(async (fileName) => {
            const fileUri = `${docDir}${fileName}`;
            const fileInfo = await FileSystem.getInfoAsync(fileUri, {
              size: true,
            });
            return {
              uri: fileUri,
              fileName,
              // Use creation/modification time or if not available, use current time minus index
              // to maintain some sort of order
              createdTime: Date.now() - videoFiles.indexOf(fileName) * 1000,
            };
          })
        );

        // Sort by creation time (oldest first)
        fileInfos.sort((a, b) => a.createdTime - b.createdTime);

        // Delete all but the MAX_LOCAL_VIDEOS most recent files
        const filesToDelete = fileInfos.slice(
          0,
          fileInfos.length - MAX_LOCAL_VIDEOS
        );

        for (const file of filesToDelete) {
          await FileSystem.deleteAsync(file.uri);
          Logger.info(`Deleted old cached video: ${file.fileName}`);
        }

        Logger.info(
          `Cleaned up ${filesToDelete.length} old videos from local storage`
        );
      }
    } catch (error) {
      Logger.error("Error cleaning up local videos", error);
    }
  };

  const downloadVideo = async (url: string) => {
    try {
      // First clean up old videos
      await cleanupLocalVideos();

      const filename = url.split("/").pop();
      const localUri = `${FileSystem.documentDirectory}${filename}`;

      Logger.info("Downloading video from URL", url);

      // Add a check to verify the URL is accessible
      const checkResponse = await fetch(url, { method: "HEAD" });
      if (!checkResponse.ok) {
        Logger.error(`URL returned status ${checkResponse.status}`, url);
        throw new Error(
          `Failed to access video URL (status ${checkResponse.status})`
        );
      }

      Logger.info("URL is accessible, downloading to", localUri);

      const downloadResult = await FileSystem.downloadAsync(url, localUri, {
        headers: {
          Accept: "video/mp4",
        },
      });

      Logger.info("Download completed with status", downloadResult.status);

      if (downloadResult.status !== 200) {
        throw new Error(`Download failed with status ${downloadResult.status}`);
      }

      // Verify the file exists and has content
      const fileInfo = await FileSystem.getInfoAsync(localUri);
      Logger.debug("Downloaded file info", fileInfo);

      if (!fileInfo.exists || fileInfo.size === 0) {
        throw new Error("Downloaded file is empty or doesn't exist");
      }

      return localUri;
    } catch (error) {
      Logger.error("Error downloading video", error);
      throw error;
    }
  };

  const pickVideo = async () => {
    try {
      setIsLoading(true);
      setError(null);
      setProcessedVideoUri(null);
      setWatermarkDetected(null);

      // Clean up any old videos before picking a new one
      await cleanupLocalVideos();

      const permissionResult =
        await ImagePicker.requestMediaLibraryPermissionsAsync();

      if (permissionResult.granted === false) {
        setError("Permission to access camera roll is required!");
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Videos,
        allowsEditing: true,
        quality: 1,
      });

      if (!result.canceled) {
        const videoUri = result.assets[0].uri;
        setVideoUri(videoUri);
        Logger.info("Selected video URI", videoUri);

        // Create form data for video upload
        const formData = new FormData();
        formData.append("video", {
          uri: videoUri,
          type: "video/mp4",
          name: "upload.mp4",
        } as any);
        formData.append("use_inpainting", useInpainting.toString());

        // Send video to backend for processing
        const backendUrl =
          process.env.EXPO_PUBLIC_BACKEND_URL || "http://localhost:5000"; // Fallback to localhost if env var not set
        Logger.info(
          "Sending video to backend for processing",
          `${backendUrl}/process-video`
        );

        const response = await fetch(`${backendUrl}/process-video`, {
          method: "POST",
          body: formData,
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        const data = await response.json();
        Logger.info("Backend response", data);

        if (data.error) {
          throw new Error(data.error);
        }

        // Set watermark detection status
        setWatermarkDetected(data.watermarkDetected);

        // Only try to download if watermark was detected and processedVideoUri exists
        if (data.watermarkDetected && data.processedVideoUri) {
          Logger.info("Backend returned video URL", data.processedVideoUri);
          try {
            // Download the video and get local URI
            const localUri = await downloadVideo(data.processedVideoUri);
            Logger.info("Video downloaded to local URI", localUri);
            setProcessedVideoUri(localUri);
          } catch (downloadError) {
            Logger.error("Failed to download video", downloadError);
            setError(
              "Failed to download the processed video. Please try again."
            );
          }
        }
      }
    } catch (error) {
      Logger.error("Error processing video", error);
      setError("Error processing video. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SafeAreaView className="flex flex-1 dark:bg-black">
      <Content
        pickVideo={pickVideo}
        videoUri={videoUri}
        watermarkDetected={watermarkDetected}
        processedVideoUri={processedVideoUri}
        isLoading={isLoading}
        error={error}
        useInpainting={useInpainting}
        setUseInpainting={setUseInpainting}
      />
    </SafeAreaView>
  );
}

function VideoPlayer({ uri }: { uri: string }) {
  Logger.debug("VideoPlayer received URI", uri);

  // For local files, we need to handle the URI format
  const videoUri = uri.startsWith("file://") ? uri : uri;
  Logger.debug("Using video URI", videoUri);

  const player = useVideoPlayer(videoUri, (player) => {
    Logger.debug("Player initialized with URI", videoUri);
    player.loop = true;
    player.play();
  });

  const { isPlaying } = useEvent(player, "playingChange", {
    isPlaying: player.playing,
  });

  return (
    <View style={styles.videoContainer}>
      <VideoView
        style={styles.video}
        player={player}
        allowsFullscreen
        allowsPictureInPicture
      />
      <View style={styles.controlsContainer}>
        <Button
          title={isPlaying ? "Pause" : "Play"}
          onPress={() => {
            if (isPlaying) {
              player.pause();
            } else {
              player.play();
            }
          }}
        />
      </View>
    </View>
  );
}

function Content({
  pickVideo,
  videoUri,
  watermarkDetected,
  processedVideoUri,
  isLoading,
  error,
  useInpainting,
  setUseInpainting,
}: ContentProps & { isLoading: boolean; error: string | null }) {
  const [saveStatus, setSaveStatus] = useState<string | null>(null);

  const saveVideo = async () => {
    try {
      if (!processedVideoUri) return;

      // Request permissions
      const { status } = await MediaLibrary.requestPermissionsAsync();
      if (status !== "granted") {
        setSaveStatus("Permission to access media library was denied");
        return;
      }

      // Save the video
      const asset = await MediaLibrary.createAssetAsync(processedVideoUri);
      await MediaLibrary.createAlbumAsync(
        "TikTok Watermark Remover",
        asset,
        false
      );
      setSaveStatus("Video saved to gallery!");
      Logger.info("Video saved to gallery", processedVideoUri);
    } catch (error) {
      Logger.error("Error saving video to gallery", error);
      setSaveStatus("Failed to save video");
    }
  };

  return (
    <View className="flex-1">
      <View>
        <View className="px-4 md:px-6">
          <View className="flex flex-col items-center text-center">
            <Text
              role="heading"
              className="text-2xl text-center native:text-4xl font-bold tracking-tighter sm:text-3xl md:text-4xl lg:text-5xl dark:text-white"
            >
              TikTok Watermark Remover
            </Text>
            <Text className="mx-auto max-w-[700px] text-lg text-center text-gray-500 md:text-xl dark:text-gray-400 mb-4">
              Upload a video to remove the TikTok watermark:
            </Text>

            <View style={[styles.toggleContainer, { marginTop: 15 }]}>
              <Text style={styles.toggleLabel}>Blur</Text>
              <Switch
                value={useInpainting}
                onValueChange={setUseInpainting}
                disabled={isLoading}
              />
              <Text style={styles.toggleLabel}>Inpainting</Text>
            </View>

            <View style={{ marginTop: 15 }}>
              <Button
                title={
                  isLoading
                    ? "Processing, this may take a few seconds."
                    : "Pick a Video"
                }
                onPress={pickVideo}
                disabled={isLoading}
              />
            </View>

            {isLoading && (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color="#4a86f7" />
              </View>
            )}

            {error && (
              <Text className="text-lg text-center text-red-500 md:text-xl">
                {error}
              </Text>
            )}

            <View style={styles.videosContainer}>
              {processedVideoUri && !isLoading && (
                <View style={styles.videoSection}>
                  <Text className="text-lg text-center text-green-500 md:text-xl">
                    Processed Video:
                  </Text>
                  <VideoPlayer uri={processedVideoUri} />
                  <View>
                    <Button title="Save to Gallery" onPress={saveVideo} />
                  </View>
                  {saveStatus && (
                    <Text className="text-sm text-center text-gray-500">
                      {saveStatus}
                    </Text>
                  )}
                </View>
              )}
            </View>

            {watermarkDetected !== null && !isLoading && (
              <View style={{ marginVertical: 10 }}>
                <Text
                  className={`text-lg text-center md:text-xl ${
                    watermarkDetected ? "text-green-500" : "text-blue-500"
                  }`}
                >
                  {watermarkDetected
                    ? "Watermark detected and removed successfully!"
                    : "No watermark detected in this video. No processing needed!"}
                </Text>
              </View>
            )}
          </View>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  videosContainer: {
    width: "100%",
    gap: 10,
  },
  videoSection: {
    width: "100%",
    alignItems: "center",
    marginVertical: 2,
    gap: 2,
  },
  videoContainer: {
    padding: 2,
    alignItems: "center",
    justifyContent: "center",
  },
  video: {
    width: 350,
    height: 275,
  },
  controlsContainer: {
    padding: 2,
  },
  toggleContainer: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginVertical: 10,
  },
  toggleLabel: {
    fontSize: 16,
    color: "#666",
  },
  loadingContainer: {
    marginTop: 20,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 10,
  },
});
