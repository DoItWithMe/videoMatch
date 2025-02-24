# Todo Parse media info
# package ffmpeg

# import (
# 	"avfp-test/internal/configs"
# 	"avfp-test/internal/utils"
# 	"context"
# 	"encoding/json"
# 	"fmt"
# 	"log"
# 	"os/exec"
# 	"time"
# )

# var (
# 	videoStreamArgs = "width,height,r_frame_rate"
# 	audioStreamArgs = "sample_fmt,sample_rate,channels"
# 	ffprobeParseCmd = "%v -i %v -loglevel error -select_streams %v -show_entries stream=codec_name,%v -show_entries format=duration -of json"
# )

# func ParseInputFileStreamInfo(ffmpegCfg *configs.FFmegConfig, filePath string, mediaType MediaType) (*StreamInfo, error) {
# 	err := utils.CheckFileExist(filePath, false)
# 	if err != nil {
# 		return nil, err
# 	}

# 	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
# 	defer cancel()

# 	// ffprobeCmdList := []string{}
# 	ffprobeCmd := ""

# 	if mediaType == VIDEO {
# 		ffprobeCmd = fmt.Sprintf(ffprobeParseCmd,
# 			ffmpegCfg.FFprobePath, filePath, "v:0", videoStreamArgs)

# 	} else if mediaType == AUDIO {
# 		ffprobeCmd = fmt.Sprintf(ffprobeParseCmd,
# 			ffmpegCfg.FFprobePath, filePath, "a:0", audioStreamArgs)
# 	} else {
# 		return nil, fmt.Errorf("mediaType should be VIDEO or AUDIO")
# 	}

# 	// ffprobeCmd := strings.Join(ffprobeCmdList, " ")

# 	cmd := exec.CommandContext(ctx, "/bin/bash", "-c", ffprobeCmd)
# 	log.Println(cmd.String())

# 	output, err := cmd.Output()
# 	if err != nil {
# 		return nil, err
# 	}

# 	streamInfo := &StreamInfo{}
# 	err = json.Unmarshal(output, streamInfo)
# 	if err != nil {
# 		return nil, err
# 	}

# 	streamInfo.Streams[0].Duration = streamInfo.Formats.Duration
# 	return streamInfo, nil
# }
