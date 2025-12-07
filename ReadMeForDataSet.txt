===============================================================================
ROBOT/DRONE TELEMETRY DATA FIELD DESCRIPTIONS
===============================================================================

This document describes each field in the telemetry dataset collected from 
a robot/drone system. The data appears to be from a ROS (Robot Operating System)
environment, capturing various sensor readings, control commands, and system states.

===============================================================================
FIELD DESCRIPTIONS
===============================================================================

--------------------------------------------------------------------------------
1. GENERAL INFORMATION
--------------------------------------------------------------------------------

S.No
    Sequential number for each data record/row


--------------------------------------------------------------------------------
2. SETPOINT RAW GLOBAL (Command Position)
--------------------------------------------------------------------------------
These fields represent the commanded target location where the robot should go.

setpoint_raw-global_Time
    Timestamp from the header of setpoint raw message

setpoint_raw-global_header.seq
    Sequence number from the header (increments with each message)

setpoint_raw-global_header.stamp.secs
    Timestamp in seconds from the header

setpoint_raw-global_latitude
    Target latitude coordinate (GPS) where robot is commanded to go

setpoint_raw-global_longitude
    Target longitude coordinate (GPS) where robot is commanded to go

setpoint_raw-global_altitude
    Target altitude where robot is commanded to go


--------------------------------------------------------------------------------
3. BATTERY STATUS
--------------------------------------------------------------------------------
Battery telemetry data from the robot's power system.

battery_Time
    Timestamp from the header of battery message

battery_header.seq
    Sequence number from the header

battery_header.stamp.secs
    Timestamp in seconds from the header

battery_voltage
    Current battery voltage (typically in volts)

battery_current
    Current being drawn from battery (typically in amperes)

battery_temperature
    Battery temperature (important for monitoring battery health)

battery_percentage
    Remaining battery charge as a percentage (0-100%)


--------------------------------------------------------------------------------
4. GLOBAL POSITION LOCAL (Current Position & Movement)
--------------------------------------------------------------------------------
The robot's current position and velocity in local coordinates.

global_position-local_Time
    Timestamp from the header

global_position-local_header.seq
    Sequence number from the header

global_position-local_header.stamp.secs
    Timestamp in seconds from the header

global_position-local_pose.pose.position.x
    Current X position of the robot in local coordinate frame

global_position-local_pose.pose.position.y
    Current Y position of the robot in local coordinate frame

global_position-local_pose.pose.position.z
    Current Z position of the robot in local coordinate frame

global_position-local_pose.pose.orientation.x
    Current X component of orientation quaternion

global_position-local_pose.pose.orientation.y
    Current Y component of orientation quaternion

global_position-local_pose.pose.orientation.z
    Current Z component of orientation quaternion

global_position-local_twist.twist.linear.x
    Current linear velocity in X direction

global_position-local_twist.twist.linear.y
    Current linear velocity in Y direction

global_position-local_twist.twist.linear.z
    Current linear velocity in Z direction


--------------------------------------------------------------------------------
5. IMU DATA (Inertial Measurement Unit)
--------------------------------------------------------------------------------
Orientation and angular velocity from the IMU sensor.

imu-data_Time
    Timestamp from the header

imu-data_header.seq
    Sequence number from the header

imu-data_header.stamp.secs
    Timestamp in seconds from the header

imu-data_orientation.x
    X component of orientation quaternion from IMU sensor

imu-data_orientation.y
    Y component of orientation quaternion from IMU sensor

imu-data_orientation.z
    Z component of orientation quaternion from IMU sensor

imu-data_orientation.w
    W component of orientation quaternion from IMU sensor

imu-data_angular_velocity.x
    Angular velocity around X axis from IMU sensor

imu-data_angular_velocity.y
    Angular velocity around Y axis from IMU sensor

imu-data_angular_velocity.z
    Angular velocity around Z axis from IMU sensor


--------------------------------------------------------------------------------
6. RC OUT (Motor Control Signals)
--------------------------------------------------------------------------------
Output signals being sent to the motors/actuators.

rc-out_Time
    Timestamp from the header

rc-out_header.seq
    Sequence number from the header

rc-out_header.stamp.secs
    Timestamp in seconds from the header

rc-out_channels_0
    Control signal to motor/actuator 1 (typically PWM value)

rc-out_channels_1
    Control signal to motor/actuator 2

rc-out_channels_2
    Control signal to motor/actuator 3

rc-out_channels_3
    Control signal to motor/actuator 4

rc-out_channels_4
    Control signal to motor/actuator 5


--------------------------------------------------------------------------------
7. VFR HUD (Vehicle Flight Report - Heads Up Display)
--------------------------------------------------------------------------------
Summary flight/movement information typically displayed to operators.

vfr_hud_Time
    Timestamp from the header

vfr_hud_header.seq
    Sequence number from the header

vfr_hud_header.stamp.secs
    Timestamp in seconds from the header

vfr_hud_airspeed
    Speed of robot relative to air (for drones/aircraft)

vfr_hud_groundspeed
    Speed of robot relative to ground

vfr_hud_heading
    Current heading/direction of travel (typically in degrees)

vfr_hud_throttle
    Current throttle setting 

vfr_hud_altitude
    Current altitude

vfr_hud_climb
    Rate of climb (positive) or descent (negative)


--------------------------------------------------------------------------------
8. GLOBAL POSITION GLOBAL (GPS Position)
--------------------------------------------------------------------------------
Robot's current position from GPS.

global_position-global_header.stamp.secs
    Timestamp in seconds from the header

global_position-global_altitude
    Current altitude from GPS

global_position-global_longitude
    Current longitude coordinate from GPS

global_position-global_latitude
    Current latitude coordinate from GPS

global_position-raw-satellites_data
    Number of GPS satellites currently connected/visible


--------------------------------------------------------------------------------
9. SETPOINT RAW TARGET GLOBAL (High-Frequency Command Position)
--------------------------------------------------------------------------------
Target position commands published at higher frequency than setpoint_raw-global.
Values are same as setpoint_raw-global but updated more frequently.

setpoint_raw-target_global_Time
    Timestamp from the header

setpoint_raw-target_global_header.seq
    Sequence number from the header

setpoint_raw-target_global_header.stamp.secs
    Timestamp in seconds from the header

setpoint_raw-target_global_latitude
    Target latitude (high-frequency update)

setpoint_raw-target_global_longitude
    Target longitude (high-frequency update)

setpoint_raw-target_global_altitude
    Target altitude (high-frequency update)

setpoint_raw-target_global_yaw
    Target yaw angle (rotation around vertical axis)


--------------------------------------------------------------------------------
10. STATE (System State Information)
--------------------------------------------------------------------------------
Overall system status and mode information.

state_Time
    Timestamp from the header

state_header.seq
    Sequence number from the header

state_connected
    Robot is connected to controller/ground station

state_armed
    Robot is armed and ready to execute commands

state_guided
    Robot is in guided mode, following setpoints autonomously

state_manual_input
    Robot is following manual input from remote control 
    (not following autonomous setpoints)

state_system_status
    General system status code/indicator


--------------------------------------------------------------------------------
11. RSSI (Received Signal Strength Indicator)
--------------------------------------------------------------------------------
Communication link quality metrics.

RSSI_Time
    Timestamp from the header

RSSI_Quality
    Quality of the received signal (typically 0-100%)

RSSI_Signal
    Strength of the received signal (typically in dBm)


--------------------------------------------------------------------------------
12. SYSTEM RESOURCES
--------------------------------------------------------------------------------
Computer/processor resource usage.

CPU_Time
    Timestamp from the header

CPU_Percent
    CPU usage percentage (0-100%)

RAM_Time
    Timestamp from the header

Used_RAM_MB
    Amount of RAM being used (in megabytes)


===============================================================================
DATA USAGE NOTES
===============================================================================

- Timestamps are typically in Unix epoch seconds
- Sequence numbers increment with each message and can help identify missing data
- Orientation is represented using quaternions (x, y, z, w components)
- Multiple sources provide similar data (e.g., position from both local and GPS)
- High-frequency fields update more rapidly than low-frequency ones
- Boolean fields (state_*) are typically 0 (false) or 1 (true)

===============================================================================
END OF FILE
===============================================================================