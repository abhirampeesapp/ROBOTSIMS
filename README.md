# RoboScout Mission Control

Full-stack robot surveillance dashboard — Flask + ROS + YOLO + Live Video.

## Quick Start

```bash
pip install -r requirements.txt
python3 app.py
# → open http://localhost:5000
```

## Architecture

```
Gazebo (TurtleBot3 Simulation)
        ↓
ROS Topics:
  /camera/rgb/image_raw   ← live frames
  /odom                   ← position + velocity
  /imu/data               ← accelerometer
  /fix                    ← GPS (NavSatFix)
  /battery_state          ← battery %
        ↓
ros_node.py  (Image subscriber → YOLO → publish /roboscout/*)
        ↓
app.py  (Flask — REST API + MJPEG streams)
        ↓
index.html  (Dashboard — cameras, map, telemetry, controls)
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/video_feed` | Raw MJPEG stream |
| GET | `/video_feed_yolo` | YOLO annotated MJPEG stream |
| GET | `/api/telemetry` | Speed, IMU, GPS, battery, CPU |
| POST | `/api/cmd_vel` | `{linear, angular}` move command |
| GET | `/api/detections` | Current YOLO detections JSON |
| POST | `/api/detections/config` | Toggle classes / intercept mode |
| POST | `/api/model/load` | Load YOLO model by name |
| GET | `/api/gps_trail` | GPS history for map trail |
| GET | `/api/ros_log` | Last 50 /rosout entries |
| POST | `/api/emergency` | Emergency stop |
| POST | `/api/emergency/clear` | Clear E-stop |
| GET | `/api/status` | ROS/camera/YOLO status |

## Running with Real ROS (TurtleBot3 + Gazebo)

```bash
# Terminal 1 — Gazebo
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_world.launch

# Terminal 2 — ROS Node (YOLO)
python3 ros_node.py _model:=yolov8n _camera_topic:=/camera/rgb/image_raw

# Terminal 3 — Flask
python3 app.py
```

## Custom Models

Place `.pt` files in `models/`:
- `models/snake_custom.pt` — activated by clicking 🐍 Snake button
- `models/fire_custom.pt`  — activated by clicking 🔥 Fire button

Train with:
```bash
yolo train data=snake.yaml model=yolov8n.pt epochs=100 imgsz=640
```

## Features

- **Dual camera view**: Raw feed left, YOLO overlay right
- **4 detection classes**: Human, Snake, Fire, Vehicle — toggle per-class
- **4 YOLO models**: YOLOv8n, YOLOv8s, Snake custom, Fire custom
- **Human intercept mode**: Robot auto-tracks detected humans using PD control
- **Auto-navigate**: Autonomous wandering with obstacle avoidance
- **GPS trail map**: Live position tracking with trail history
- **Full telemetry**: Speed, angular velocity, heading, battery, CPU, IMU, GPS
- **Emergency stop**: Hardware-level E-stop + web button + keyboard Space
- **Keyboard control**: Arrow keys for movement
- **ROS log feed**: Live /rosout scrolling display
- **Simulation mode**: Works without ROS/camera using synthetic test patterns
