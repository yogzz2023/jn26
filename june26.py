import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
import mplcursors
from scipy.stats import chi2

# Constants
GATE_SIZE = 9.21  # Chi-square value for 95% confidence interval with 2 DOF

r=[]
el=[]
az=[]

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time
        print("Initialized filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.Pf)

    def InitializeMeasurementForFiltering(self, x, y, z, vx, vy, vz, mt):
        Z = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = mt

    def predict_step(self, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        print("Predicted filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.Pf)

    def update_step(self):
        # Update step with JPDA
        Inn = self.Z - np.dot(self.H, self.Sf)  # Calculate innovation directly
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)
        print("Updated filter state:")
        print("Sf:", self.Sf)
        print("pf:", self.Pf)

class Track:
    def __init__(self, id, initial_state, initial_time):
        self.id = id
        self.kalman_filter = CVFilter()
        self.kalman_filter.initialize_filter_state(*initial_state, initial_time)
        self.last_update_time = initial_time

    def predict(self, current_time):
        self.kalman_filter.predict_step(current_time)

    def update(self, measurement):
        self.kalman_filter.Z = measurement
        self.kalman_filter.update_step()
        self.last_update_time = measurement[-1]

    def get_state(self):
        return self.kalman_filter.Sf

# Helper function to calculate Mahalanobis distance
def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    try:
        inv_cov = np.linalg.inv(cov)
        return np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff)).item()
    except np.linalg.LinAlgError:
        return np.inf  # Return a large distance if covariance matrix is singular or non-invertible

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

# Function to convert Cartesian coordinates to spherical coordinates
def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z / np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x) * 180 / np.pi
    if az < 0:
        az += 360
    return r, az, el

def cart2sph2(x: float, y: float, z: float, filtered_values_csv):
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2))
        el.append(math.atan(z[i] / np.sqrt(x[i] ** 2 + y[i] ** 2)) * 180 / 3.14)
        az.append(math.atan(y[i] / x[i]))

        if x[i] > 0.0:
            az[i] = 3.14 / 2 - az[i]
        else:
            az[i] = 3 * 3.14 / 2 - az[i]

        az[i] = az[i] * 180 / 3.14

        if az[i] < 0.0:
            az[i] = (360 + az[i])

        if az[i] > 360:
            az[i] = (az[i] - 360)

    return r, az, el

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            print("Cartesian coordinates (x, y, z):", x, y, z)
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            print("Spherical coordinates (r, az, el):", r, az, el)
            measurements.append((r, az, el, mt))
    return measurements

# Function to initiate new tracks
def initiate_tracks(measurements, tracks, current_time, next_track_id):
    new_tracks = []
    for measurement in measurements:
        state = (measurement[0], measurement[1], measurement[2], 0, 0, 0)
        new_track = Track(next_track_id, state, current_time)
        new_tracks.append(new_track)
        next_track_id += 1
    tracks.extend(new_tracks)
    return next_track_id

# Chi-Square Test for Measurement-to-Track Association
def chi_square_test(measurement, track, gate_size):
    track_state = track.get_state()[:3]
    measurement_cov = track.kalman_filter.R
    distance = mahalanobis_distance(np.array(measurement[:3]).reshape(-1, 1), track_state, measurement_cov)
    return distance < gate_size

# Forming Clusters and Hypotheses
def form_clusters(measurements, tracks):
    clusters = []
    for measurement in measurements:
        cluster = []
        for track in tracks:
            if chi_square_test(measurement, track, GATE_SIZE):
                cluster.append(track)
        clusters.append((measurement, cluster))
    return clusters

def form_hypotheses(clusters):
    hypotheses = []
    for measurement, cluster in clusters:
        if cluster:
            for track in cluster:
                hypotheses.append((measurement, track))
        else:
            hypotheses.append((measurement, None))  # New track initiation
    return hypotheses

def calculate_joint_probabilities(hypotheses):
    joint_probabilities = []
    for measurement, track in hypotheses:
        if track:
            likelihood = 1.0  # Placeholder, compute likelihood based on measurement and track state
        else:
            likelihood = 0.1  # Placeholder for new track initiation
        joint_probabilities.append(likelihood)
    # Normalize probabilities
    total_prob = sum(joint_probabilities)
    joint_probabilities = [prob / total_prob for prob in joint_probabilities]
    return joint_probabilities

def jpda_update(measurements, tracks, current_time):
    clusters = form_clusters(measurements, tracks)
    hypotheses = form_hypotheses(clusters)
    joint_probabilities = calculate_joint_probabilities(hypotheses)

    for prob, (measurement, track) in zip(joint_probabilities, hypotheses):
        if track:
            track.kalman_filter.Z = np.array(measurement[:3]).reshape(-1, 1)
            track.kalman_filter.update_step()
        else:
            # Initiate a new track with the measurement
            next_track_id = initiate_tracks([measurement], tracks, current_time, next_track_id)

def process_measurements(measurements, tracks, next_track_id):
    for i, (r, az, el, mt) in enumerate(measurements):
        if i == 0:
            # Initialize first track
            state = (r, az, el, 0, 0, 0)
            next_track_id = initiate_tracks([state], tracks, mt, next_track_id)
        else:
            # Predict step for all tracks
            for track in tracks:
                track.predict(mt)
            # JPDA update step
            jpda_update([(r, az, el, mt)], tracks, mt)

# Example usage
tracks = []
next_track_id = 1
measurements = read_measurements_from_csv('ttk_84.csv')
process_measurements(measurements, tracks, next_track_id)

# Convert Cartesian coordinates back to spherical
filtered_values_csv = measurements  # Assuming measurements is your filtered data
cartesian_coords = [(x, y, z) for (r, az, el, mt) in measurements for (x, y, z) in [sph2cart(az, el, r)]]
x_vals, y_vals, z_vals = zip(*cartesian_coords)
r_vals, az_vals, el_vals = cart2sph2(x_vals, y_vals, z_vals, filtered_values_csv)

# Prepare DataFrame for plotting
data = {'r': r_vals, 'az': az_vals, 'el': el_vals}
df = pd.DataFrame(data)

# Plot results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['az'], df['el'], df['r'], c='r', marker='o')

ax.set_xlabel('Azimuth')
ax.set_ylabel('Elevation')
ax.set_zlabel('Range')

mplcursors.cursor(hover=True)
plt.show()
