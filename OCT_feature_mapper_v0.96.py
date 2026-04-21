from __future__ import annotations

import csv
import json
import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import shutil
import tempfile
import zipfile

try:
    import pydicom
except ImportError:
    pydicom = None


import cv2
import numpy as np
import tkinter as tk
from shapely.geometry import Polygon
from tkinter import filedialog, messagebox, ttk


DISPLAY_SCALE = 3.0
DEFAULT_INTENSITY = 100
DEFAULT_FILTER_MODE = 0
WINDOW_BSCAN = "BSCAN"
WINDOW_POLYGON = "Manual Feature Polygon"
WINDOW_BSCAN_HELP = "BSCAN HELP"
WINDOW_POLYGON_HELP = "POLYGON HELP"
APP_TITLE = "OCT feature mapper v0.96"
MAX_FEATURE_POINTS_PER_BSCAN = 6
HEADER_HEIGHT = 90
BOX_MARGIN_X = 20
BOX_MARGIN_Y = 14
BOX_WIDTH = 262
BOX_HEIGHT = 61
PREVIEW_PANEL_GAP = 18
PREVIEW_PANEL_WIDTH_RATIO = 0.62
PREVIEW_MAX_HEIGHT = 1200
PREVIEW_MIN_WIDTH = 260
PREVIEW_MAX_WIDTH = 1200
PREVIEW_LINE_COLOUR = (190, 190, 190)
PREVIEW_POINT_COLOUR = (255, 0, 0)
PREVIEW_CURRENT_POINT_COLOUR = (0, 0, 255)
PREVIEW_IMAGE_FILL_PANEL = False
POLYGON_PANEL_WIDTH = 310
POLYGON_PANEL_PADDING = 16
POLYGON_BUTTON_HEIGHT = 56
POLYGON_BUTTON_GAP = 12
POLYGON_BUTTON_FILL = (54, 54, 58)
POLYGON_BUTTON_BORDER = (168, 168, 172)
POLYGON_BUTTON_HOVER_FILL = (72, 72, 78)
POLYGON_BUTTON_HOVER_BORDER = (214, 214, 220)
POLYGON_BUTTON_PRESSED_FILL = (34, 34, 38)
POLYGON_BUTTON_PRESSED_BORDER = (136, 136, 140)
POLYGON_BUTTON_PRESSED_OFFSET = 3
POLYGON_PANEL_BACKGROUND = (26, 27, 30)
POLYGON_PANEL_TEXT = (238, 238, 238)
POLYGON_PANEL_MUTED = (168, 168, 168)
COREG_BLEND_ALPHA = 0.75
ENFACE_DISPLAY_UPSCALE = 1.55


@dataclass
class OCTMetadata:
    scale_x_enface: float
    scale_x_bscan: float
    scale_y_enface: float
    res_x: float
    start_x: float
    y_coords: List[float]
    plot_start_x: float
    plot_end_x: float
    plot_start_y: float
    plot_end_y: float
    enface_file: str
    xml_file: str


@dataclass
class SessionResult:
    folder: str
    xml_file: str
    enface_file: str
    bscan_count: int
    labelled_bscan_count: int
    feature_coords_by_bscan: dict[int, List[int]] = field(default_factory=dict)
    polygon_vertices_px: List[Tuple[int, int]] = field(default_factory=list)
    points_px_crop: List[Tuple[int, int]] = field(default_factory=list)
    feature_area_mm2: Optional[float] = None
    feature_perimeter_mm: Optional[float] = None
    polygon_centroids_px: List[Tuple[float, float]] = field(default_factory=list)
    polygon_centroids_mm: List[Tuple[float, float]] = field(default_factory=list)
    overlay_image: Optional[str] = None
    csv_file: Optional[str] = None
    measurements_file: Optional[str] = None
    overlay_opened: bool = False
    cancelled_by_user: bool = False
    polygon_count: int = 0


@dataclass
class PolygonMeasurement:
    area_mm2: float
    perimeter_mm: float
    centroid_px: Tuple[float, float]
    centroid_mm: Tuple[float, float]


class OCTFeatureAreaWorkflow:
    def __init__(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
        ui_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.progress_callback = progress_callback or (lambda message: None)
        self.cancel_callback = cancel_callback or (lambda: False)
        self.ui_callback = ui_callback or (lambda: None)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.clahe_on = False
        self.selected_vertices: List[Tuple[int, int]] = []
        self.completed_polygons: List[List[Tuple[int, int]]] = []
        self.current_points: List[Tuple[int, int]] = []
        self.feature_coords: dict[int, List[int]] = {}
        self.feature_click_points_by_bscan: dict[int, List[Tuple[int, int]]] = {}
        self._current_bscan_original: Optional[np.ndarray] = None
        self._bscan_index = 0
        self._current_display_y_offset = HEADER_HEIGHT
        self._metadata: Optional[OCTMetadata] = None
        self._bscans: List[np.ndarray] = []
        self._folder: Optional[Path] = None
        self._xml_path: Optional[Path] = None
        self._points_px_crop: List[Tuple[int, int]] = []
        self._area: Optional[float] = None
        self._overlay_output_name: Optional[str] = None
        self._csv_output_name: Optional[str] = None
        self._v1 = self._v2 = self._h1 = self._h2 = 0
        self._enface_full: Optional[np.ndarray] = None
        self._clahe_box_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._bscan_box_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._overlay_requested = False
        self._quit_requested = False
        self._last_bscan_index = 0
        self._temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._workspace_root: Optional[Path] = None
        self._oct_dataset = None
        self._localizer_dataset = None
        self._bscan_frame_order: List[int] = []
        self._frame_reference_coords_px: List[Tuple[float, float, float, float]] = []
        self._overlay_crop_bounds_px: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._points_px_full: List[Tuple[int, int]] = []
        self._overlay_tool: Optional[str] = None
        self._line_marker_points: List[Tuple[int, int]] = []
        self._completed_line_markers: List[List[Tuple[int, int]]] = []
        self._fovea_point: Optional[Tuple[int, int]] = None
        self._fovea_bscan_index: Optional[int] = None
        self._fovea_bscan_point: Optional[Tuple[int, int]] = None
        self._bscan_fovea_mode = False
        self._point_features: List[Tuple[int, int]] = []
        self._pending_bscan_step = 0
        self._hover_bscan_x: Optional[int] = None
        self._polygon_button_rects: dict[str, Tuple[int, int, int, int]] = {}
        self._polygon_pending_action: Optional[str] = None
        self._polygon_hover_action: Optional[str] = None
        self._polygon_button_pressed_until: dict[str, float] = {}
        self._coregistered_enface_full: Optional[np.ndarray] = None
        self._coregistration_image_name: Optional[str] = None
        self._coregistration_status: str = "No image loaded"
        self._coregistration_uploaded_canvas: Optional[np.ndarray] = None
        self._coregistration_transform: Optional[np.ndarray] = None
        self._coregistration_auto_transform: Optional[np.ndarray] = None
        self._coregistration_base_status: str = "No image loaded"
        self._coregistration_manual_adjusted = False
        self._coregistration_drag_active = False
        self._coregistration_drag_moved = False
        self._coregistration_drag_start_display: Optional[Tuple[int, int]] = None
        self._coregistration_drag_start_transform: Optional[np.ndarray] = None
        self._overlay_display_scale_x = 1.0
        self._overlay_display_scale_y = 1.0

    def report(self, message: str) -> None:
        self.progress_callback(message)

    def _should_cancel(self) -> bool:
        try:
            self.ui_callback()
        except Exception:
            pass
        return bool(self.cancel_callback())

    def show_save_confirmation(self, message: str) -> None:
        try:
            root = tk._default_root
            if root is not None:
                root.update_idletasks()
            messagebox.showinfo(APP_TITLE, message)
        except Exception:
            try:
                messagebox.showinfo(APP_TITLE, message)
            except Exception:
                pass

    def run(self, xml_path: Path) -> SessionResult:
        try:
            self._xml_path = xml_path
            self._folder = xml_path.parent
            self._overlay_requested = False
            self._quit_requested = False
            self._last_bscan_index = 0
            self._reset_overlay_state()
            self.report(f"Loading OCT folder: {self._folder}")
            self._metadata = self.load_metadata(xml_path)
            self.report(f"Loaded DICOM metadata: {self._metadata.xml_file}")
            self._bscans = self.load_bscans(self._folder)
            self.report(f"Loaded {len(self._bscans)} embedded B-scan images")

            stage = "bscan"
            while True:
                if stage == "bscan":
                    action = self.run_bscan_labelling(start_idx=self._last_bscan_index)
                    if action == "quit":
                        self._quit_requested = True
                        self.report("Measurement cancelled by user and returned to the main GUI.")
                        return self.save_session_outputs()
                    if action == "overlay":
                        self._overlay_requested = True
                        stage = "polygon"
                        continue
                    stage = "bscan"

                if stage == "polygon":
                    action = self.run_polygon_overlay()
                    if action == "quit":
                        self._quit_requested = True
                        self.report("Measurement cancelled by user and returned to the main GUI.")
                        return self.save_session_outputs()
                    if action == "back":
                        self.report("Returned from enface overlay to B-scan editing.")
                        stage = "bscan"
                        continue
                    break

            result = self.save_session_outputs()
            if result.feature_area_mm2 is not None:
                self.report(f"Finished. Feature area = {result.feature_area_mm2:.3f} mm^2")
            else:
                self.report("Finished. Session saved without a final polygon area.")
            return result
        finally:
            cv2.destroyAllWindows()
            if self._temp_dir is not None:
                self._temp_dir.cleanup()
                self._temp_dir = None

    def show_instructions(self, title: str, lines: List[str], size: Tuple[int, int] = (230, 480)) -> None:
        img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
        y = 24
        for line in lines:
            cv2.putText(img, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
            y += 22
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(title, img)
        cv2.waitKey(1)
        try:
            cv2.moveWindow(title, 1060, 30)
        except cv2.error:
            pass

    def _safe_destroy_window(self, title: str) -> None:
        try:
            cv2.destroyWindow(title)
        except cv2.error:
            pass


    def _require_pydicom(self) -> None:
        if pydicom is None:
            raise ImportError(
                "The DICOM version of OCT feature mapper requires the 'pydicom' package. "
                "Install it with: pip install pydicom"
            )

    def _prepare_dicom_workspace(self, source_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
        if source_path.suffix.lower() == ".zip":
            temp_dir = tempfile.TemporaryDirectory(prefix="oct_feature_mapper_dicom_")
            with zipfile.ZipFile(source_path, "r") as zf:
                zf.extractall(temp_dir.name)
            return Path(temp_dir.name), temp_dir
        if source_path.is_dir():
            return source_path, None
        return source_path.parent, None

    def _find_dicom_files(self, root: Path) -> List[Path]:
        if root.is_file():
            return [root]
        files: List[Path] = []
        for path in root.rglob("*"):
            if path.is_file() and path.name.upper() != "DICOMDIR":
                files.append(path)
        return sorted(files)

    def _read_dicom_dataset(self, path: Path, stop_before_pixels: bool = False):
        self._require_pydicom()
        return pydicom.dcmread(str(path), stop_before_pixels=stop_before_pixels, force=True)

    def _dataset_laterality(self, ds) -> str:
        for attr in ("Laterality", "ImageLaterality"):
            value = getattr(ds, attr, "")
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
        return ""

    def _image_type_tokens(self, ds) -> List[str]:
        value = getattr(ds, "ImageType", [])
        if isinstance(value, str):
            return [value.upper()]
        return [str(item).upper() for item in value]

    def _pixel_spacing_xy(self, ds) -> Tuple[float, float]:
        pixel_spacing = getattr(ds, "PixelSpacing", None)
        if pixel_spacing and len(pixel_spacing) >= 2:
            return float(pixel_spacing[1]), float(pixel_spacing[0])
        shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
        if shared:
            for group in shared:
                measures = getattr(group, "PixelMeasuresSequence", None)
                if measures:
                    spacing = getattr(measures[0], "PixelSpacing", None)
                    if spacing and len(spacing) >= 2:
                        return float(spacing[1]), float(spacing[0])
        raise ValueError("Could not determine DICOM pixel spacing.")

    def _reference_coords(self, frame_group) -> Optional[Tuple[float, float, float, float]]:
        seq = getattr(frame_group, "OphthalmicFrameLocationSequence", None)
        if seq:
            ref = getattr(seq[0], "ReferenceCoordinates", None)
            if ref and len(ref) >= 4:
                vals = [float(v) for v in ref[:4]]
                return vals[0], vals[1], vals[2], vals[3]
        return None

    def _normalize_for_cv2(self, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                min_val = float(arr.min())
                max_val = float(arr.max())
                if max_val > min_val:
                    arr = (arr - min_val) * (255.0 / (max_val - min_val))
                else:
                    arr = np.zeros_like(arr)
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if arr.ndim == 2:
            arr = arr.astype(np.float32)
            min_val = float(arr.min())
            max_val = float(arr.max())
            if max_val > min_val:
                arr = (arr - min_val) * (255.0 / (max_val - min_val))
            else:
                arr = np.zeros_like(arr)
            arr8 = np.clip(arr, 0, 255).astype(np.uint8)
            return cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[0] == 1:
            return self._normalize_for_cv2(arr[0])
        raise ValueError(f"Unsupported DICOM pixel array shape: {arr.shape}")

    def load_metadata(self, xml_path: Path) -> OCTMetadata:
        self._require_pydicom()
        source_path = xml_path
        workspace, temp_dir = self._prepare_dicom_workspace(source_path)
        self._temp_dir = temp_dir
        self._workspace_root = workspace
        dicom_files = self._find_dicom_files(workspace)
        if not dicom_files:
            raise FileNotFoundError("No DICOM files were found in the selected export.")

        oct_candidates = []
        localizer_candidates = []
        for path in dicom_files:
            try:
                ds = self._read_dicom_dataset(path, stop_before_pixels=True)
            except Exception:
                continue
            frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
            rows = int(getattr(ds, "Rows", 0) or 0)
            cols = int(getattr(ds, "Columns", 0) or 0)
            tokens = self._image_type_tokens(ds)
            entry = (path, ds, frames, rows, cols, tokens)
            if frames > 1:
                oct_candidates.append(entry)
            elif "LOCALIZER" in tokens or rows == cols:
                localizer_candidates.append(entry)

        if not oct_candidates:
            raise ValueError("No multi-frame OCT volume was found in the selected DICOM export.")

        oct_candidates.sort(key=lambda item: (item[2], item[3] * item[4]), reverse=True)
        oct_path, oct_ds, _, _, _, _ = oct_candidates[0]
        laterality = self._dataset_laterality(oct_ds)

        matching_localizers = [entry for entry in localizer_candidates if self._dataset_laterality(entry[1]) == laterality]
        if not matching_localizers:
            matching_localizers = localizer_candidates
        if not matching_localizers:
            raise ValueError("No localizer/enface DICOM image was found in the selected export.")
        matching_localizers.sort(key=lambda item: item[3] * item[4], reverse=True)
        localizer_path, localizer_ds, _, _, _, _ = matching_localizers[0]

        oct_full = self._read_dicom_dataset(oct_path, stop_before_pixels=False)
        localizer_full = self._read_dicom_dataset(localizer_path, stop_before_pixels=False)
        self._oct_dataset = oct_full
        self._localizer_dataset = localizer_full

        enface_scale_x, enface_scale_y = self._pixel_spacing_xy(localizer_full)
        bscan_scale_x, _ = self._pixel_spacing_xy(oct_full)
        res_x = float(getattr(oct_full, "Columns", 0) or 0)
        if res_x <= 0:
            raise ValueError("Could not determine B-scan width from DICOM.")

        frame_groups = getattr(oct_full, "PerFrameFunctionalGroupsSequence", None)
        if not frame_groups:
            raise ValueError("The OCT DICOM volume does not contain per-frame location metadata.")

        frame_positions: List[Tuple[int, float, float, float, float, float, float, float]] = []
        for index, frame_group in enumerate(frame_groups):
            ref = self._reference_coords(frame_group)
            if ref is None:
                raise ValueError("The OCT DICOM volume is missing OphthalmicFrameLocationSequence reference coordinates.")
            x1_px, y1_px, x2_px, y2_px = ref
            x_start_mm = min(x1_px, x2_px) * enface_scale_x
            x_end_mm = max(x1_px, x2_px) * enface_scale_x
            y_mid_px = (float(y1_px) + float(y2_px)) * 0.5
            y_mm = y_mid_px * enface_scale_y
            frame_positions.append((index, x_start_mm, y_mm, x_end_mm, float(x1_px), float(y1_px), float(x2_px), float(y2_px)))

        frame_positions.sort(key=lambda item: item[2], reverse=True)
        self._bscan_frame_order = [item[0] for item in frame_positions]
        self._frame_reference_coords_px = [(item[4], item[5], item[6], item[7]) for item in frame_positions]
        y_coords = [item[2] for item in frame_positions]
        start_x = min(item[1] for item in frame_positions)
        end_x = max(item[3] for item in frame_positions)

        start_y_coord = y_coords[0]
        end_y_coord = y_coords[-1]
        no_bscans = len(y_coords)
        scale_lat_y = (start_y_coord - end_y_coord) / (no_bscans - 1) if no_bscans > 1 else 0.0

        plot_start_x = start_x - (0.5 * bscan_scale_x)
        plot_end_x = end_x + (0.5 * bscan_scale_x)
        plot_start_y = start_y_coord + (0.5 * scale_lat_y)
        plot_end_y = end_y_coord - (0.5 * scale_lat_y)

        xs = [coord for item in self._frame_reference_coords_px for coord in (item[0], item[2])]
        y_mids_px = [0.5 * (item[1] + item[3]) for item in self._frame_reference_coords_px]
        if len(y_mids_px) > 1:
            sorted_y = sorted(y_mids_px, reverse=True)
            y_steps = [abs(sorted_y[i] - sorted_y[i + 1]) for i in range(len(sorted_y) - 1) if abs(sorted_y[i] - sorted_y[i + 1]) > 0]
            pad_y = 0.5 * min(y_steps) if y_steps else 0.5
        else:
            pad_y = 0.5
        self._overlay_crop_bounds_px = (
            max(0, int(np.floor(min(xs)))),
            int(np.ceil(max(xs))),
            max(0, int(np.floor(min(y_mids_px) - pad_y))),
            int(np.ceil(max(y_mids_px) + pad_y)),
        )

        localizer_pixels = np.asarray(localizer_full.pixel_array)
        if localizer_pixels.ndim == 3 and localizer_pixels.shape[0] == 1:
            localizer_pixels = localizer_pixels[0]
        self._enface_full = self._normalize_for_cv2(localizer_pixels)

        self.report(
            f"Loaded DICOM volume ({len(self._bscan_frame_order)} B-scans, laterality {laterality or 'unknown'}) "
            f"from {source_path.name}"
        )

        return OCTMetadata(
            scale_x_enface=enface_scale_x,
            scale_x_bscan=bscan_scale_x,
            scale_y_enface=enface_scale_y,
            res_x=res_x,
            start_x=start_x,
            y_coords=y_coords,
            plot_start_x=plot_start_x,
            plot_end_x=plot_end_x,
            plot_start_y=plot_start_y,
            plot_end_y=plot_end_y,
            enface_file=localizer_path.name,
            xml_file=source_path.name,
        )

    def load_bscans(self, folder: Path) -> List[np.ndarray]:
        if not hasattr(self, "_oct_dataset"):
            raise ValueError("DICOM OCT volume has not been loaded.")
        pixel_array = np.asarray(self._oct_dataset.pixel_array)
        if pixel_array.ndim != 3:
            raise ValueError(f"Expected a multi-frame OCT volume, got pixel array shape {pixel_array.shape}")
        bscans: List[np.ndarray] = []
        for frame_index in self._bscan_frame_order:
            bscans.append(self._normalize_for_cv2(pixel_array[frame_index]))
        return bscans

    def _decode_mouse_wheel(self, flags: int) -> int:
        delta = (flags >> 16) & 0xFFFF
        if delta >= 0x8000:
            delta -= 0x10000
        return delta

    def on_bscan_click(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_MOUSEWHEEL:
            wheel_delta = self._decode_mouse_wheel(flags)
            if wheel_delta > 0:
                self._pending_bscan_step = 1
            elif wheel_delta < 0:
                self._pending_bscan_step = -1
            return

        if event == cv2.EVENT_MOUSEMOVE:
            x_orig = int(x / DISPLAY_SCALE)
            y_orig = int((y - self._current_display_y_offset) / DISPLAY_SCALE)
            if (
                self._current_bscan_original is not None
                and 0 <= y_orig < self._current_bscan_original.shape[0]
                and 0 <= x_orig < self._current_bscan_original.shape[1]
            ):
                self._hover_bscan_x = x_orig
            else:
                self._hover_bscan_x = None
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        x1, y1, x2, y2 = self._clahe_box_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.clahe_on = not self.clahe_on
            return

        x_orig = int(x / DISPLAY_SCALE)
        y_orig = int((y - self._current_display_y_offset) / DISPLAY_SCALE)
        if y_orig < 0:
            return

        if self._bscan_fovea_mode:
            self._set_bscan_fovea(self._bscan_index, x_orig, y_orig)
            self._bscan_fovea_mode = False
            return

        if len(self.current_points) >= MAX_FEATURE_POINTS_PER_BSCAN:
            return
        self.current_points.append((x_orig, y_orig))

    def _store_current_points_for_bscan(self, idx: int) -> bool:
        if len(self.current_points) >= 1:
            saved_points = [tuple(point) for point in self.current_points[:MAX_FEATURE_POINTS_PER_BSCAN]]
            self.feature_click_points_by_bscan[idx + 1] = saved_points
            self.feature_coords[idx + 1] = sorted([point[0] for point in saved_points])
            self.report(f"Saved {len(saved_points)} feature marks for B-scan {idx + 1}")
            return True
        return False

    def _commit_current_bscan_before_overlay(self, idx: int) -> None:
        stored = self._store_current_points_for_bscan(idx)
        if not stored and (idx + 1) in self.feature_click_points_by_bscan:
            saved_points = [tuple(point) for point in self.feature_click_points_by_bscan[idx + 1][:MAX_FEATURE_POINTS_PER_BSCAN]]
            self.current_points = saved_points
            self.feature_coords[idx + 1] = sorted([point[0] for point in saved_points])

    def run_bscan_labelling(self, start_idx: int = 0) -> str:
        self.report("Manual B-scan labelling started")
        cv2.namedWindow(WINDOW_BSCAN, cv2.WINDOW_NORMAL)
        try:
            cv2.getTrackbarPos("Intensity x100", WINDOW_BSCAN)
        except cv2.error:
            cv2.createTrackbar("Intensity x100", WINDOW_BSCAN, DEFAULT_INTENSITY, 200, lambda x: None)
            cv2.createTrackbar("Colour Filter", WINDOW_BSCAN, DEFAULT_FILTER_MODE, 4, lambda x: None)
        cv2.setMouseCallback(WINDOW_BSCAN, self.on_bscan_click)
        self.show_instructions(
            WINDOW_BSCAN_HELP,
            [
                "Left click = Mark feature points (up to 6)",
                "v then click = Mark fovea on B-scan",
                "Mouse wheel = Next / previous B-scan",
                "b = Previous B-scan",
                "u = Clear current clicks",
                "f = Next B-scan",
                "o = Open enface overlay",
                "q = Quit to main GUI",
                "Click CLAHE box = Toggle contrast enhancement",
            ],
        )

        idx = max(0, min(start_idx, len(self._bscans) - 1))
        total = len(self._bscans)
        while True:
            self._current_bscan_original = self._bscans[idx].copy()
            self._bscan_index = idx
            self._last_bscan_index = idx
            self._hover_bscan_x = None
            self.current_points = [tuple(point) for point in self.feature_click_points_by_bscan.get(idx + 1, [])]
            self.report(f"B-scan {idx + 1} of {total}")

            while True:
                if self._should_cancel():
                    self._quit_requested = True
                    self._safe_destroy_window(WINDOW_BSCAN)
                    self._safe_destroy_window(WINDOW_BSCAN_HELP)
                    self.report("Measurement cancelled by Exit button.")
                    return "quit"
                display = self.prepare_bscan_display(self._current_bscan_original)
                cv2.imshow(WINDOW_BSCAN, display)
                key = cv2.waitKey(20) & 0xFF

                if self._pending_bscan_step != 0:
                    if len(self.current_points) >= 1:
                        self._store_current_points_for_bscan(idx)
                    idx = (idx + self._pending_bscan_step) % total
                    self._pending_bscan_step = 0
                    break

                if key == ord("b"):
                    if len(self.current_points) >= 1:
                        self._store_current_points_for_bscan(idx)
                    idx = (idx - 1) % total
                    break
                elif key == ord("u"):
                    self.current_points = []
                    self._bscan_fovea_mode = False
                elif key == ord("f"):
                    if len(self.current_points) >= 1:
                        self._store_current_points_for_bscan(idx)
                    idx = (idx + 1) % total
                    break
                elif key == ord("v"):
                    self._bscan_fovea_mode = True
                    self.report(f"Fovea marking mode: click the fovea on B-scan {idx + 1}.")
                elif key == ord("o"):
                    self._commit_current_bscan_before_overlay(idx)
                    self._last_bscan_index = idx
                    self._safe_destroy_window(WINDOW_BSCAN)
                    self._safe_destroy_window(WINDOW_BSCAN_HELP)
                    cv2.waitKey(1)
                    self.report("Opening enface overlay from the current measurement state.")
                    return "overlay"
                elif key == ord("q"):
                    self._last_bscan_index = idx
                    self._safe_destroy_window(WINDOW_BSCAN)
                    self._safe_destroy_window(WINDOW_BSCAN_HELP)
                    self.report("B-scan labelling closed by user and returned to the main GUI.")
                    return "quit"

    def _map_full_to_overlay(self, x_full: float, y_full: float) -> Tuple[int, int]:
        return int(round(y_full)), int(round(x_full))

    def _project_bscan_x_to_overlay(self, ref_coords: Tuple[float, float, float, float], xpix: int) -> Tuple[int, int]:
        assert self._metadata is not None
        x1_px, y1_px, x2_px, y2_px = ref_coords
        bscan_width = max(int(round(self._metadata.res_x)) - 1, 1)
        frac = max(0.0, min(1.0, float(xpix) / float(bscan_width)))
        x_full = x1_px + frac * (x2_px - x1_px)
        y_full = y1_px + frac * (y2_px - y1_px)
        return self._map_full_to_overlay(x_full, y_full)

    def _set_bscan_fovea(self, idx: int, xpix: int, ypix: int) -> None:
        if idx < 0 or idx >= len(self._frame_reference_coords_px):
            return
        self._fovea_bscan_index = idx
        self._fovea_bscan_point = (int(xpix), int(ypix))
        self._fovea_point = self._project_bscan_x_to_overlay(self._frame_reference_coords_px[idx], int(xpix))
        self.report(
            f"Fovea marked on B-scan {idx + 1}: B-scan x={xpix}, y={ypix}; "
            f"enface x={self._fovea_point[0]}, y={self._fovea_point[1]} px."
        )

    def _collect_preview_points(self) -> List[Tuple[int, int, bool]]:
        preview_points: List[Tuple[int, int, bool]] = []
        for idx_key, xvals in sorted(self.feature_coords.items()):
            ref_index = idx_key - 1
            if ref_index < 0 or ref_index >= len(self._frame_reference_coords_px):
                continue
            ref_coords = self._frame_reference_coords_px[ref_index]
            for xpix in xvals:
                x_map, y_map = self._project_bscan_x_to_overlay(ref_coords, xpix)
                preview_points.append((x_map, y_map, False))

        if 0 <= self._bscan_index < len(self._frame_reference_coords_px):
            ref_coords = self._frame_reference_coords_px[self._bscan_index]
            for point in self.current_points:
                x_map, y_map = self._project_bscan_x_to_overlay(ref_coords, int(point[0]))
                preview_points.append((x_map, y_map, True))
        return preview_points

    def _hover_preview_position(self) -> Optional[Tuple[int, int]]:
        if self._hover_bscan_x is None:
            return None
        if 0 <= self._bscan_index < len(self._frame_reference_coords_px):
            ref_coords = self._frame_reference_coords_px[self._bscan_index]
            return self._project_bscan_x_to_overlay(ref_coords, self._hover_bscan_x)
        return None

    def _build_live_enface_preview(self, target_height: int, bscan_display_width: int) -> np.ndarray:
        panel_w = int(round(bscan_display_width * PREVIEW_PANEL_WIDTH_RATIO))
        panel_w = max(PREVIEW_MIN_WIDTH, min(panel_w, PREVIEW_MAX_WIDTH))
        panel_h = max(target_height, 160)

        panel = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 18
        cv2.putText(panel, "Live enface", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        if self._enface_full is None:
            return panel

        preview = self._enface_full.copy()
        h, w = preview.shape[:2]

        if 0 <= self._bscan_index < len(self._frame_reference_coords_px):
            x1_px, y1_px, x2_px, y2_px = self._frame_reference_coords_px[self._bscan_index]
            pt1 = self._map_full_to_overlay(x1_px, y1_px)
            pt2 = self._map_full_to_overlay(x2_px, y2_px)
            cv2.line(preview, pt1, pt2, PREVIEW_LINE_COLOUR, 1, cv2.LINE_AA)

        hover_position = self._hover_preview_position()
        if hover_position is not None:
            hover_x = max(0, min(int(round(hover_position[0])), w - 1))
            hover_y = max(0, min(int(round(hover_position[1])), h - 1))
            cv2.circle(preview, (hover_x, hover_y), 3, PREVIEW_LINE_COLOUR, -1, cv2.LINE_AA)

        for x_map, y_map, is_current in self._collect_preview_points():
            x_map = max(0, min(int(round(x_map)), w - 1))
            y_map = max(0, min(int(round(y_map)), h - 1))
            colour = PREVIEW_CURRENT_POINT_COLOUR if is_current else PREVIEW_POINT_COLOUR
            radius = 4 if is_current else 3
            cv2.circle(preview, (x_map, y_map), radius, colour, -1, cv2.LINE_AA)

        if self._fovea_point is not None:
            fx = max(0, min(int(round(self._fovea_point[0])), w - 1))
            fy = max(0, min(int(round(self._fovea_point[1])), h - 1))
            cv2.drawMarker(
                preview,
                (fx, fy),
                (0, 255, 255),
                markerType=cv2.MARKER_STAR,
                markerSize=12,
                thickness=1,
                line_type=cv2.LINE_AA,
            )

        preview = self._resize_enface_for_display(preview)
        h, w = preview.shape[:2]

        top_pad = 30
        inner_pad = 6
        fit_h = max(panel_h - top_pad - inner_pad, 40)
        fit_w = max(panel_w - (2 * inner_pad), 40)

        scale = min(fit_w / float(w), fit_h / float(h))
        scale = max(scale, 0.1)
        resized = cv2.resize(
            preview,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4,
        )

        x_off = max((panel_w - resized.shape[1]) // 2, 0)
        y_off = top_pad + max((fit_h - resized.shape[0]) // 2, 0)
        x_end = min(x_off + resized.shape[1], panel_w)
        y_end = min(y_off + resized.shape[0], panel_h)
        panel[y_off:y_end, x_off:x_end] = resized[: y_end - y_off, : x_end - x_off]
        return panel

    def prepare_bscan_display(self, img_original: np.ndarray) -> np.ndarray:
        intensity = cv2.getTrackbarPos("Intensity x100", WINDOW_BSCAN) / 100.0
        filter_mode = cv2.getTrackbarPos("Colour Filter", WINDOW_BSCAN)

        gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        if self.clahe_on:
            gray = self.clahe.apply(gray)
        gray = cv2.convertScaleAbs(gray, alpha=intensity, beta=0)

        if filter_mode == 0:
            img_proc = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif filter_mode == 1:
            img_proc = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        elif filter_mode == 2:
            img_proc = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        elif filter_mode == 3:
            img_proc = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
        else:
            img_proc = cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)

        img_display = cv2.resize(
            img_proc,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_LANCZOS4,
        )
        display_width = img_display.shape[1]
        header = np.zeros((self._current_display_y_offset, display_width, 3), dtype=np.uint8)
        display = np.vstack([header, img_display])

        clahe_x1 = BOX_MARGIN_X
        clahe_y1 = BOX_MARGIN_Y
        clahe_x2 = min(clahe_x1 + BOX_WIDTH, display_width - BOX_MARGIN_X)
        clahe_y2 = min(clahe_y1 + BOX_HEIGHT, self._current_display_y_offset - 8)
        self._clahe_box_rect = (clahe_x1, clahe_y1, clahe_x2, clahe_y2)

        bscan_x2 = display_width - BOX_MARGIN_X
        bscan_y1 = BOX_MARGIN_Y
        bscan_x1 = max(bscan_x2 - BOX_WIDTH, BOX_MARGIN_X + 10)
        bscan_y2 = min(bscan_y1 + BOX_HEIGHT, self._current_display_y_offset - 8)
        self._bscan_box_rect = (bscan_x1, bscan_y1, bscan_x2, bscan_y2)

        clahe_fill = (30, 180, 65) if self.clahe_on else (70, 70, 70)
        clahe_border = (170, 255, 190) if self.clahe_on else (180, 180, 180)
        cv2.rectangle(display, (clahe_x1, clahe_y1), (clahe_x2, clahe_y2), clahe_fill, -1)
        cv2.rectangle(display, (clahe_x1, clahe_y1), (clahe_x2, clahe_y2), clahe_border, 2)
        cv2.putText(display, "CLAHE", (clahe_x1 + 16, clahe_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)
        cv2.putText(
            display,
            "ON" if self.clahe_on else "OFF",
            (clahe_x1 + 16, clahe_y1 + 49),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.76,
            (255, 255, 255),
            2,
        )

        cv2.rectangle(display, (bscan_x1, bscan_y1), (bscan_x2, bscan_y2), (52, 52, 52), -1)
        cv2.rectangle(display, (bscan_x1, bscan_y1), (bscan_x2, bscan_y2), (210, 210, 210), 2)
        cv2.putText(display, "B-scan", (bscan_x1 + 16, bscan_y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(
            display,
            f"{self._bscan_index + 1}/{len(self._bscans)}",
            (bscan_x1 + 16, bscan_y1 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
        )
        if self._bscan_fovea_mode:
            cv2.putText(
                display,
                "FOVEA MODE: click fovea",
                (max(clahe_x2 + 20, 20), 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for pt in self.current_points:
            x_disp = int(pt[0] * DISPLAY_SCALE)
            y_disp = int(pt[1] * DISPLAY_SCALE) + self._current_display_y_offset
            cv2.circle(display, (x_disp, y_disp), 6, (0, 0, 255), -1)

        if self._fovea_bscan_index == self._bscan_index and self._fovea_bscan_point is not None:
            fx_disp = int(self._fovea_bscan_point[0] * DISPLAY_SCALE)
            fy_disp = int(self._fovea_bscan_point[1] * DISPLAY_SCALE) + self._current_display_y_offset
            cv2.drawMarker(
                display,
                (fx_disp, fy_disp),
                (0, 255, 255),
                markerType=cv2.MARKER_STAR,
                markerSize=24,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.circle(display, (fx_disp, fy_disp), 8, (0, 255, 255), 2, cv2.LINE_AA)

        preview_panel = self._build_live_enface_preview(display.shape[0], display.shape[1])
        gap = np.zeros((display.shape[0], PREVIEW_PANEL_GAP, 3), dtype=np.uint8)
        combined = np.hstack([display, gap, preview_panel])
        return combined

    def build_projected_points(self) -> List[Tuple[int, int]]:
        points_px: List[Tuple[int, int]] = []
        for idx_key, xvals in sorted(self.feature_coords.items()):
            ref_index = idx_key - 1
            if ref_index < 0 or ref_index >= len(self._frame_reference_coords_px):
                continue
            ref_coords = self._frame_reference_coords_px[ref_index]
            for xpix in xvals:
                points_px.append(self._project_bscan_x_to_overlay(ref_coords, xpix))
        return points_px

    def _enface_display_scale_factors(self) -> Tuple[float, float]:
        if self._metadata is None:
            return ENFACE_DISPLAY_UPSCALE, ENFACE_DISPLAY_UPSCALE
        base_scale = min(self._metadata.scale_x_enface, self._metadata.scale_y_enface)
        if base_scale <= 0:
            return ENFACE_DISPLAY_UPSCALE, ENFACE_DISPLAY_UPSCALE
        return (
            (self._metadata.scale_x_enface / base_scale) * ENFACE_DISPLAY_UPSCALE,
            (self._metadata.scale_y_enface / base_scale) * ENFACE_DISPLAY_UPSCALE,
        )

    def _resize_enface_for_display(self, image: np.ndarray) -> np.ndarray:
        fx, fy = self._enface_display_scale_factors()
        self._overlay_display_scale_x = fx
        self._overlay_display_scale_y = fy
        if abs(fx - 1.0) < 1e-6 and abs(fy - 1.0) < 1e-6:
            return image
        return cv2.resize(
            image,
            None,
            fx=fx,
            fy=fy,
            interpolation=cv2.INTER_LINEAR,
        )

    def _truncate_label(self, text: str, max_chars: int = 26) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _reset_overlay_state(self) -> None:
        self.selected_vertices = []
        self.completed_polygons = []
        self._line_marker_points = []
        self._completed_line_markers = []
        self._point_features = []
        self._overlay_tool = None
        self._polygon_pending_action = None
        self._polygon_hover_action = None
        self._polygon_button_pressed_until = {}
        self._area = None
        self._coregistered_enface_full = None
        self._coregistration_image_name = None
        self._coregistration_status = "No image loaded"
        self._coregistration_uploaded_canvas = None
        self._coregistration_transform = None
        self._coregistration_auto_transform = None
        self._coregistration_base_status = "No image loaded"
        self._coregistration_manual_adjusted = False
        self._coregistration_drag_active = False
        self._coregistration_drag_moved = False
        self._coregistration_drag_start_display = None
        self._coregistration_drag_start_transform = None

    def _build_polygon_control_panel(self, panel_height: int) -> np.ndarray:
        panel = np.full((panel_height, POLYGON_PANEL_WIDTH, 3), POLYGON_PANEL_BACKGROUND, dtype=np.uint8)
        self._polygon_button_rects = {}
        now = time.monotonic()

        cv2.putText(
            panel,
            "Overlay Tools",
            (POLYGON_PANEL_PADDING, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.90,
            POLYGON_PANEL_TEXT,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "Click a button or use keys.",
            (POLYGON_PANEL_PADDING, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            POLYGON_PANEL_MUTED,
            1,
            cv2.LINE_AA,
        )

        button_specs = [
            ("draw_polygon", "Draw Polygon", "Select polygon tool", False),
            ("next_polygon", "Next Polygon", "Finish current polygon", False),
            ("line_marker", "Line Marker", "Click sequential line points", False),
            ("next_line_marker", "Next Line Marker", "Finish current line marker", False),
            ("point_feature", "Point Feature", "Mark feature point", False),
            ("delete_points", "Delete Points", "Click points to remove them", True),
            ("back_to_bscans", "Back to Bscans", "Return to B-scan editing", False),
            ("coregister", "Coregister Image", "Browse + align image", False),
            ("save", "Save", "Save overlay + CSV", False),
        ]

        button_y = 102
        for key, label, subtitle, add_section_gap in button_specs:
            x1 = POLYGON_PANEL_PADDING
            x2 = POLYGON_PANEL_WIDTH - POLYGON_PANEL_PADDING
            y1 = button_y
            y2 = y1 + POLYGON_BUTTON_HEIGHT
            self._polygon_button_rects[key] = (x1, y1, x2, y2)
            is_active_tool = (
                (key == "draw_polygon" and self._overlay_tool == "polygon")
                or (key == "line_marker" and self._overlay_tool == "line")
                or (key == "point_feature" and self._overlay_tool == "point")
                or (key == "delete_points" and self._overlay_tool == "delete")
            )
            is_pressed = self._polygon_button_pressed_until.get(key, 0.0) > now
            is_hovered = self._polygon_hover_action == key and not is_pressed
            fill = (
                POLYGON_BUTTON_PRESSED_FILL
                if is_pressed
                else (38, 96, 72)
                if is_active_tool
                else POLYGON_BUTTON_HOVER_FILL
                if is_hovered
                else POLYGON_BUTTON_FILL
            )
            border = (
                POLYGON_BUTTON_PRESSED_BORDER
                if is_pressed
                else (112, 238, 178)
                if is_active_tool
                else POLYGON_BUTTON_HOVER_BORDER
                if is_hovered
                else POLYGON_BUTTON_BORDER
            )
            y_offset = POLYGON_BUTTON_PRESSED_OFFSET if is_pressed else 0
            if not is_pressed:
                cv2.rectangle(panel, (x1 + 2, y1 + 4), (x2 + 2, y2 + 4), (18, 18, 20), -1)
            cv2.rectangle(panel, (x1, y1 + y_offset), (x2, y2 + y_offset), fill, -1)
            cv2.rectangle(panel, (x1, y1 + y_offset), (x2, y2 + y_offset), border, 2)
            cv2.putText(
                panel,
                label,
                (x1 + 16, y1 + y_offset + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                POLYGON_PANEL_TEXT,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                subtitle,
                (x1 + 16, y1 + y_offset + 47),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                POLYGON_PANEL_MUTED,
                1,
                cv2.LINE_AA,
            )
            button_y = y2 + POLYGON_BUTTON_GAP
            if add_section_gap:
                button_y += POLYGON_BUTTON_HEIGHT // 2

        info_y = button_y + 18
        active_tool_text = "None selected"
        if self._overlay_tool == "polygon":
            active_tool_text = "Draw Polygon"
        elif self._overlay_tool == "line":
            active_tool_text = "Line Marker"
        elif self._overlay_tool == "point":
            active_tool_text = "Point Feature"
        elif self._overlay_tool == "delete":
            active_tool_text = "Delete Points"
        info_lines = [
            "Active Tool",
            active_tool_text,
            "",
            "Shortcuts",
            "p = Draw polygon",
            "n = Next polygon",
            "l = Line marker",
            "m = Next line",
            "x = Point feature",
            "d = Delete points",
            "b = Back to B-scans",
            "c = Coregister image",
            "s = Save",
            "Drag = Move coreg",
            "j/k = Rotate",
            "-/= = Scale",
            "0 = Reset coreg",
            "q = Finish session",
            "",
            "Coregistration",
            self._truncate_label(self._coregistration_image_name or "No image loaded"),
            self._truncate_label(self._coregistration_status),
        ]
        for line in info_lines:
            if line == "":
                info_y += 10
                continue
            colour = POLYGON_PANEL_TEXT if line in {"Active Tool", "Shortcuts", "Coregistration"} else POLYGON_PANEL_MUTED
            thickness = 2 if colour == POLYGON_PANEL_TEXT else 1
            font_scale = 0.58 if thickness == 2 else 0.47
            cv2.putText(panel, line, (POLYGON_PANEL_PADDING, info_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, thickness, cv2.LINE_AA)
            info_y += 24
        return panel

    def _render_polygon_overlay_base(self) -> np.ndarray:
        assert self._enface_full is not None
        overlay = self._enface_full.copy()
        if self._coregistered_enface_full is not None:
            overlay = cv2.addWeighted(overlay, 1.0 - COREG_BLEND_ALPHA, self._coregistered_enface_full, COREG_BLEND_ALPHA, 0.0)
        return overlay

    def _start_next_polygon(self) -> bool:
        if len(self.selected_vertices) < 3:
            self.report("Need at least 3 points before starting the next polygon.")
            return False
        self.completed_polygons.append(self.selected_vertices.copy())
        self.selected_vertices = []
        self.report(f"Started next polygon. Completed polygons: {len(self.completed_polygons)}")
        return True

    def _start_next_line_marker(self) -> bool:
        if len(self._line_marker_points) < 2:
            self.report("Need at least 2 points before starting the next line marker.")
            return False
        self._completed_line_markers.append(self._line_marker_points.copy())
        self._line_marker_points = []
        self.report(f"Started next line marker. Completed line markers: {len(self._completed_line_markers)}")
        return True

    def _save_overlay_snapshot(self, overlay: np.ndarray, notify: bool = True) -> bool:
        if self._folder is None or self._metadata is None:
            self.report("No output folder is available yet.")
            return False
        has_line_marker = bool(self._line_marker_points) or bool(self._completed_line_markers)
        if self._area is None and not has_line_marker and self._fovea_point is None and not self._point_features:
            self.report("No polygon area, line marker, fovea point, or point feature is available to save yet.")
            return False
        if self._coregistered_enface_full is not None:
            self._overlay_output_name = "coregistered_feature_overlay.png"
        else:
            self._overlay_output_name = "feature_overlay.png"
        cv2.imwrite(str(self._folder / self._overlay_output_name), overlay)
        self.report(f"Saved overlay image: {self._overlay_output_name}")
        self.save_session_outputs()
        if notify:
            self.show_save_confirmation(
                f"Images and coordinates saved in\n{self._folder}\n\nDICOM file:\n{self._metadata.xml_file}"
            )
        return True

    def _odd_kernel_size(self, value: float, minimum: int = 3) -> int:
        size = max(minimum, int(round(value)))
        if size % 2 == 0:
            size += 1
        return size

    def _compose_affine(self, dest_from_mid: np.ndarray, mid_from_src: np.ndarray) -> np.ndarray:
        dest_from_mid_h = np.vstack([dest_from_mid.astype(np.float32), [0.0, 0.0, 1.0]])
        mid_from_src_h = np.vstack([mid_from_src.astype(np.float32), [0.0, 0.0, 1.0]])
        return (dest_from_mid_h @ mid_from_src_h)[:2].astype(np.float32)

    def _project_affine_to_similarity(self, matrix: np.ndarray) -> np.ndarray:
        affine = matrix.astype(np.float32).copy()
        a, b, tx = [float(v) for v in affine[0]]
        c, d, ty = [float(v) for v in affine[1]]
        scale_x = math.hypot(a, c)
        scale_y = math.hypot(b, d)
        scale = max(1e-4, min(3.0, 0.5 * (scale_x + scale_y)))
        theta = math.atan2(c - b, a + d)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        return np.array(
            [
                [scale * cos_t, -scale * sin_t, tx],
                [scale * sin_t, scale * cos_t, ty],
            ],
            dtype=np.float32,
        )

    def _centered_similarity_delta(
        self,
        image_shape: Tuple[int, int],
        translation_x: float = 0.0,
        translation_y: float = 0.0,
        rotation_deg: float = 0.0,
        scale_factor: float = 1.0,
    ) -> np.ndarray:
        h, w = image_shape[:2]
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
        delta = cv2.getRotationMatrix2D(center, rotation_deg, scale_factor).astype(np.float32)
        delta[0, 2] += float(translation_x)
        delta[1, 2] += float(translation_y)
        return delta

    def _build_coregistration_canvas(self, image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        if src_h <= 0 or src_w <= 0:
            raise ValueError("Invalid uploaded image size for coregistration.")
        scale = min(target_w / float(src_w), target_h / float(src_h))
        scaled_w = max(1, int(round(src_w * scale)))
        scaled_h = max(1, int(round(src_h * scale)))
        resized = cv2.resize(
            image,
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_AREA if src_h * src_w >= target_h * target_w else cv2.INTER_LANCZOS4,
        )
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        offset_x = (target_w - scaled_w) // 2
        offset_y = (target_h - scaled_h) // 2
        canvas[offset_y:offset_y + scaled_h, offset_x:offset_x + scaled_w] = resized
        return canvas

    def _retinal_gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image.copy()
        else:
            bgr = image.astype(np.uint8)
            green = bgr[:, :, 1]
            luminance = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.addWeighted(luminance, 0.35, green, 0.65, 0.0)
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def _build_fovea_bias(self, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape[:2]
        yy, xx = np.indices((h, w), dtype=np.float32)
        center_x = (w - 1) * 0.5
        center_y = (h - 1) * 0.5
        sigma = max(12.0, 0.22 * float(min(h, w)))
        bias = np.exp(-(((xx - center_x) ** 2) + ((yy - center_y) ** 2)) / (2.0 * sigma * sigma))
        return cv2.normalize(bias, None, 0.2, 1.0, cv2.NORM_MINMAX).astype(np.float32)

    def _prepare_coregistration_features(self, image: np.ndarray):
        gray = self._retinal_gray(image)
        h, w = gray.shape[:2]
        min_dim = float(min(h, w))

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        background_sigma = max(4.0, min_dim * 0.02)
        background = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=background_sigma, sigmaY=background_sigma)
        local_contrast = cv2.normalize(cv2.absdiff(enhanced, background), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        vessel_response = np.zeros_like(enhanced, dtype=np.float32)
        for frac in (0.012, 0.02, 0.03):
            kernel_size = self._odd_kernel_size(min_dim * frac, minimum=5)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dark_ridges = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
            bright_ridges = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
            np.maximum(vessel_response, dark_ridges.astype(np.float32), out=vessel_response)
            np.maximum(vessel_response, bright_ridges.astype(np.float32), out=vessel_response)

        sobel_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = cv2.magnitude(sobel_x, sobel_y)
        vessel_u8 = cv2.normalize(0.8 * vessel_response + 0.2 * gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vessel_u8 = cv2.GaussianBlur(vessel_u8, (0, 0), sigmaX=1.0, sigmaY=1.0)

        vessel_threshold = max(18.0, float(np.percentile(vessel_u8, 72.0)))
        vessel_mask = np.where(vessel_u8 >= vessel_threshold, 255, 0).astype(np.uint8)
        clean_size = self._odd_kernel_size(min_dim * 0.006, minimum=3)
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clean_size, clean_size))
        vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, clean_kernel)
        vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_CLOSE, clean_kernel)

        center_bias = self._build_fovea_bias(gray.shape)
        center_radius = max(12, int(round(min_dim * 0.46)))
        center_mask = np.zeros_like(vessel_mask)
        cv2.circle(center_mask, (w // 2, h // 2), center_radius, 255, -1)
        content_mask = np.where(gray > 3, 255, 0).astype(np.uint8)
        expand_size = self._odd_kernel_size(min_dim * 0.01, minimum=5)
        expand_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_size, expand_size))
        focus_mask = cv2.dilate(vessel_mask, expand_kernel)
        central_focus_mask = cv2.bitwise_and(focus_mask, center_mask)
        if cv2.countNonZero(central_focus_mask) >= max(200, int(0.015 * h * w)):
            focus_mask = central_focus_mask
        focus_mask = cv2.bitwise_and(focus_mask, content_mask)
        feature_mix = (
            0.88 * vessel_u8.astype(np.float32)
            + 0.12 * local_contrast.astype(np.float32)
        ) * center_bias
        feature_u8 = cv2.normalize(feature_mix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return {
            "gray": gray,
            "feature": feature_u8,
            "vessel_mask": vessel_mask,
            "focus_mask": focus_mask,
            "center_bias": center_bias,
        }

    def _weighted_correlation(self, fixed_image: np.ndarray, moving_image: np.ndarray, weight: np.ndarray) -> float:
        weights = weight.astype(np.float32)
        valid = weights > 1e-4
        if int(np.count_nonzero(valid)) < 64:
            return -1.0
        fixed_vals = fixed_image[valid].astype(np.float32)
        moving_vals = moving_image[valid].astype(np.float32)
        weight_vals = weights[valid]
        fixed_mean = float(np.average(fixed_vals, weights=weight_vals))
        moving_mean = float(np.average(moving_vals, weights=weight_vals))
        fixed_zero = fixed_vals - fixed_mean
        moving_zero = moving_vals - moving_mean
        denom = math.sqrt(
            float(np.sum(weight_vals * fixed_zero * fixed_zero))
            * float(np.sum(weight_vals * moving_zero * moving_zero))
        )
        if denom <= 1e-8:
            return -1.0
        return float(np.sum(weight_vals * fixed_zero * moving_zero) / denom)

    def _estimate_similarity_from_phase_search(
        self,
        fixed_feature: np.ndarray,
        moving_feature: np.ndarray,
        fixed_mask: Optional[np.ndarray],
        moving_mask: Optional[np.ndarray],
        center_bias: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        h, w = fixed_feature.shape[:2]
        fixed_float = fixed_feature.astype(np.float32) / 255.0
        moving_float = moving_feature.astype(np.float32) / 255.0
        fixed_support = np.ones((h, w), dtype=np.float32)
        if fixed_mask is not None:
            fixed_support = np.maximum(fixed_mask.astype(np.float32) / 255.0, 0.0)
        moving_support = np.ones((h, w), dtype=np.float32)
        if moving_mask is not None:
            moving_support = np.maximum(moving_mask.astype(np.float32) / 255.0, 0.0)

        base_window = fixed_support * center_bias
        reference_area = max(64, int(np.count_nonzero(base_window > 0.05) * 0.18))
        best_score = -1e9
        best_matrix = None

        scales = (0.88, 0.94, 1.0, 1.06, 1.12)
        angles = (-18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0)
        for scale in scales:
            for angle in angles:
                candidate = self._centered_similarity_delta((h, w), rotation_deg=angle, scale_factor=scale)
                rotated_feature = cv2.warpAffine(
                    moving_float,
                    candidate,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                rotated_support = cv2.warpAffine(
                    moving_support,
                    candidate,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                window = base_window * rotated_support
                if int(np.count_nonzero(window > 0.05)) < reference_area:
                    continue

                shift, response = cv2.phaseCorrelate(fixed_float, rotated_feature, window=window)
                trial = candidate.copy()
                trial[0, 2] += float(shift[0])
                trial[1, 2] += float(shift[1])

                aligned_feature = cv2.warpAffine(
                    moving_float,
                    trial,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                aligned_support = cv2.warpAffine(
                    moving_support,
                    trial,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                eval_weight = base_window * aligned_support
                if int(np.count_nonzero(eval_weight > 0.05)) < reference_area:
                    continue

                corr = self._weighted_correlation(fixed_float, aligned_feature, eval_weight)
                overlap_ratio = float(np.mean(eval_weight > 0.05))
                score = (0.65 * float(response)) + (0.30 * corr) + (0.05 * overlap_ratio)
                if score > best_score:
                    best_score = score
                    best_matrix = trial.astype(np.float32)

        return best_matrix, float(best_score)

    def _apply_coregistration_transform(self) -> bool:
        if self._enface_full is None or self._coregistration_uploaded_canvas is None or self._coregistration_transform is None:
            return False
        target_h, target_w = self._enface_full.shape[:2]
        self._coregistration_transform = self._project_affine_to_similarity(self._coregistration_transform)
        self._coregistered_enface_full = cv2.warpAffine(
            self._coregistration_uploaded_canvas,
            self._coregistration_transform,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        if self._coregistration_manual_adjusted:
            self._coregistration_status = f"{self._coregistration_base_status} + manual refine"
        else:
            self._coregistration_status = self._coregistration_base_status
        return True

    def _adjust_coregistration(
        self,
        translation_x: float = 0.0,
        translation_y: float = 0.0,
        rotation_deg: float = 0.0,
        scale_factor: float = 1.0,
    ) -> bool:
        if self._coregistration_transform is None or self._enface_full is None:
            return False
        delta = self._centered_similarity_delta(
            self._enface_full.shape[:2],
            translation_x=translation_x,
            translation_y=translation_y,
            rotation_deg=rotation_deg,
            scale_factor=scale_factor,
        )
        self._coregistration_transform = self._compose_affine(delta, self._coregistration_transform)
        self._coregistration_manual_adjusted = True
        return self._apply_coregistration_transform()

    def _reset_coregistration_adjustment(self) -> bool:
        if self._coregistration_auto_transform is None:
            return False
        self._coregistration_transform = self._coregistration_auto_transform.copy()
        self._coregistration_manual_adjusted = False
        return self._apply_coregistration_transform()

    def _overlay_display_to_full_point(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        if self._enface_full is None:
            return None
        x_image = x - POLYGON_PANEL_WIDTH
        if x_image < 0:
            return None
        x_unscaled = float(x_image) / max(self._overlay_display_scale_x, 1e-6)
        y_unscaled = float(y) / max(self._overlay_display_scale_y, 1e-6)
        height, width = self._enface_full.shape[:2]
        x_full = max(0, min(int(round(x_unscaled)), width - 1))
        y_full = max(0, min(int(round(y_unscaled)), height - 1))
        return x_full, y_full

    def _polygon_anchor_for_click(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        full_point = self._overlay_display_to_full_point(x, y)
        if full_point is None:
            return None

        x_full, y_full = full_point
        snap_threshold_px = 12.0 / max(min(self._overlay_display_scale_x, self._overlay_display_scale_y), 1e-6)
        snap_candidates: List[Tuple[float, Tuple[int, int], str]] = []

        for point in self._points_px_full:
            distance = float(np.hypot(x_full - point[0], y_full - point[1]))
            snap_candidates.append((distance, point, "bscan_marker"))

        if snap_candidates:
            distance, point, _source = min(snap_candidates, key=lambda item: item[0])
            if distance <= snap_threshold_px:
                return point
        return x_full, y_full

    def _append_polygon_vertex(self, x: int, y: int) -> None:
        point = self._polygon_anchor_for_click(x, y)
        if point is None:
            return
        x_full, y_full = point
        self.selected_vertices.append((x_full, y_full))

    def _append_line_marker_point(self, x: int, y: int) -> None:
        point = self._polygon_anchor_for_click(x, y)
        if point is None:
            return
        self._line_marker_points.append(point)
        self.report(f"Line marker point {len(self._line_marker_points)} added.")

    def _append_point_feature(self, x: int, y: int) -> None:
        point = self._overlay_display_to_full_point(x, y)
        if point is None:
            return
        self._point_features.append(point)
        self.report(f"Point feature {len(self._point_features)} marked at x={point[0]}, y={point[1]} px.")

    def _delete_nearest_overlay_point(self, x: int, y: int) -> None:
        point = self._overlay_display_to_full_point(x, y)
        if point is None:
            return
        x_full, y_full = point
        delete_threshold_px = 14.0 / max(min(self._overlay_display_scale_x, self._overlay_display_scale_y), 1e-6)
        candidates: List[Tuple[float, str, int, int]] = []

        for idx, feature in enumerate(self._point_features):
            candidates.append((float(np.hypot(x_full - feature[0], y_full - feature[1])), "point_feature", idx, -1))
        for idx, vertex in enumerate(self.selected_vertices):
            candidates.append((float(np.hypot(x_full - vertex[0], y_full - vertex[1])), "current_polygon", idx, -1))
        for poly_idx, polygon in enumerate(self.completed_polygons):
            for vertex_idx, vertex in enumerate(polygon):
                candidates.append((float(np.hypot(x_full - vertex[0], y_full - vertex[1])), "completed_polygon", poly_idx, vertex_idx))
        for idx, vertex in enumerate(self._line_marker_points):
            candidates.append((float(np.hypot(x_full - vertex[0], y_full - vertex[1])), "current_line", idx, -1))
        for line_idx, line_points in enumerate(self._completed_line_markers):
            for point_idx, line_point in enumerate(line_points):
                candidates.append((float(np.hypot(x_full - line_point[0], y_full - line_point[1])), "completed_line", line_idx, point_idx))

        if not candidates:
            self.report("No point features, polygon points, or line marker points are available to delete.")
            return

        distance, kind, outer_idx, inner_idx = min(candidates, key=lambda item: item[0])
        if distance > delete_threshold_px:
            self.report("Click closer to a point feature, polygon point, or line marker point to delete it.")
            return

        if kind == "point_feature":
            self._point_features.pop(outer_idx)
            self.report("Deleted point feature.")
            return

        if kind == "current_polygon":
            self.selected_vertices.pop(outer_idx)
            self.report("Deleted point from current polygon.")
            return

        if kind == "completed_polygon":
            polygon = self.completed_polygons[outer_idx]
            polygon.pop(inner_idx)
            if len(polygon) < 3:
                self.completed_polygons.pop(outer_idx)
                self.report("Deleted point and removed polygon because fewer than 3 points remained.")
            else:
                self.report("Deleted point from completed polygon.")
            return

        if kind == "current_line":
            self._line_marker_points.pop(outer_idx)
            self.report("Deleted point from current line marker.")
            return

        if kind == "completed_line":
            line_points = self._completed_line_markers[outer_idx]
            line_points.pop(inner_idx)
            if len(line_points) < 2:
                self._completed_line_markers.pop(outer_idx)
                self.report("Deleted point and removed line marker because fewer than 2 points remained.")
            else:
                self.report("Deleted point from completed line marker.")
            return

    def _handle_overlay_image_click(self, x: int, y: int) -> None:
        if self._overlay_tool == "polygon":
            self._append_polygon_vertex(x, y)
        elif self._overlay_tool == "line":
            self._append_line_marker_point(x, y)
        elif self._overlay_tool == "point":
            self._append_point_feature(x, y)
        elif self._overlay_tool == "delete":
            self._delete_nearest_overlay_point(x, y)
        else:
            self.report("Select Draw Polygon, Line Marker, Point Feature, or Delete Points before marking the enface overlay.")

    def _undo_active_overlay_mark(self) -> None:
        if self._overlay_tool == "point":
            if self._point_features:
                self._point_features.pop()
            return
        if self._overlay_tool == "line":
            if self._line_marker_points:
                self._line_marker_points.pop()
            elif self.selected_vertices:
                self.selected_vertices.pop()
            return
        if self.selected_vertices:
            self.selected_vertices.pop()
        elif self._line_marker_points:
            self._line_marker_points.pop()

    def _reset_active_overlay_marks(self) -> None:
        if self._overlay_tool == "point":
            self._point_features = []
        elif self._overlay_tool == "line":
            self._line_marker_points = []
        else:
            self.selected_vertices = []

    def _all_line_marker_paths(self, include_current: bool = True) -> List[List[Tuple[int, int]]]:
        paths = [line.copy() for line in self._completed_line_markers if line]
        if include_current and self._line_marker_points:
            paths.append(self._line_marker_points.copy())
        return paths

    def _line_marker_length_mm(self, line_points: Optional[List[Tuple[int, int]]] = None) -> float:
        if self._metadata is None:
            return 0.0
        if line_points is None:
            return float(sum(self._line_marker_length_mm(points) for points in self._all_line_marker_paths(include_current=True)))
        if len(line_points) < 2:
            return 0.0
        total = 0.0
        for start_point, end_point in zip(line_points, line_points[1:]):
            dx_mm = float(end_point[0] - start_point[0]) * self._metadata.scale_x_enface
            dy_mm = float(end_point[1] - start_point[1]) * self._metadata.scale_y_enface
            total += math.hypot(dx_mm, dy_mm)
        return float(total)

    def _line_marker_label_position(self, line_points: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not line_points:
            return 30, 80
        x_vals = [point[0] for point in line_points]
        y_vals = [point[1] for point in line_points]
        return int(round(sum(x_vals) / len(x_vals))) + 10, int(round(sum(y_vals) / len(y_vals))) - 10

    def _estimate_similarity_from_keypoints(
        self,
        fixed_feature: np.ndarray,
        moving_feature: np.ndarray,
        fixed_mask: Optional[np.ndarray],
        moving_mask: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], int]:
        detector = cv2.AKAZE_create()
        fixed_keypoints, fixed_descriptors = detector.detectAndCompute(fixed_feature, fixed_mask)
        moving_keypoints, moving_descriptors = detector.detectAndCompute(moving_feature, moving_mask)
        if (
            fixed_descriptors is None
            or moving_descriptors is None
            or len(fixed_keypoints) < 6
            or len(moving_keypoints) < 6
        ):
            return None, 0

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn_matches = matcher.knnMatch(moving_descriptors, fixed_descriptors, k=2)
        good_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.78 * second.distance:
                good_matches.append(first)

        if len(good_matches) < 6:
            return None, len(good_matches)

        moving_pts = np.float32([moving_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        fixed_pts = np.float32([fixed_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        moving_to_fixed, inliers = cv2.estimateAffinePartial2D(
            moving_pts,
            fixed_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.995,
            refineIters=50,
        )
        if moving_to_fixed is None:
            return None, len(good_matches)
        if inliers is not None and int(inliers.sum()) < 5:
            return None, len(good_matches)
        return moving_to_fixed.astype(np.float32), len(good_matches)

    def _coregister_uploaded_image(self) -> bool:
        assert self._enface_full is not None
        image_path = filedialog.askopenfilename(
            title="Select image to coregister",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not image_path:
            self.report("Coregistration image selection cancelled.")
            return False

        uploaded = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if uploaded is None:
            messagebox.showerror(APP_TITLE, f"Could not open image:\n{image_path}")
            self.report(f"Coregistration failed to open image: {image_path}")
            return False

        target_h, target_w = self._enface_full.shape[:2]
        uploaded_resized = cv2.resize(
            uploaded,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA if uploaded.shape[0] * uploaded.shape[1] >= target_h * target_w else cv2.INTER_LANCZOS4,
        )
        fixed_gray = cv2.cvtColor(self._enface_full, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        moving_gray = cv2.cvtColor(uploaded_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            100,
            1e-5,
        )
        mode_text = "Resized overlay fallback"
        moving_to_fixed = np.eye(2, 3, dtype=np.float32)
        try:
            cv2.findTransformECC(
                fixed_gray,
                moving_gray,
                warp_matrix,
                cv2.MOTION_AFFINE,
                criteria,
            )
            moving_to_fixed = self._project_affine_to_similarity(cv2.invertAffineTransform(warp_matrix))
            mode_text = "ECC affine alignment"
        except cv2.error as exc:
            self.report(f"Coregistration fallback used for {Path(image_path).name}: {exc}")

        self._coregistration_uploaded_canvas = uploaded_resized
        self._coregistration_auto_transform = moving_to_fixed.copy()
        self._coregistration_transform = moving_to_fixed.copy()
        self._coregistration_manual_adjusted = False
        self._coregistration_drag_active = False
        self._coregistration_drag_moved = False
        self._coregistration_drag_start_display = None
        self._coregistration_drag_start_transform = None
        self._coregistration_image_name = Path(image_path).name
        self._coregistration_base_status = mode_text
        self._coregistration_status = self._coregistration_base_status
        self._apply_coregistration_transform()
        self.report(f"Coregistered image loaded: {self._coregistration_image_name} ({self._coregistration_status})")
        return True

    def on_polygon_click(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        hovered_action = None
        for action, rect in self._polygon_button_rects.items():
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                hovered_action = action
                break

        if event == cv2.EVENT_MOUSEMOVE:
            self._polygon_hover_action = hovered_action
            if self._coregistration_drag_active and self._coregistration_drag_start_display is not None and self._coregistration_drag_start_transform is not None:
                start_x, start_y = self._coregistration_drag_start_display
                dx_display = float(x - start_x)
                dy_display = float(y - start_y)
                dx_full = dx_display / max(self._overlay_display_scale_x, 1e-6)
                dy_full = dy_display / max(self._overlay_display_scale_y, 1e-6)
                if abs(dx_full) >= 0.5 or abs(dy_full) >= 0.5:
                    delta = np.array([[1.0, 0.0, dx_full], [0.0, 1.0, dy_full]], dtype=np.float32)
                    self._coregistration_transform = self._compose_affine(delta, self._coregistration_drag_start_transform)
                    self._coregistration_manual_adjusted = True
                    self._coregistration_drag_moved = True
                    self._apply_coregistration_transform()
            return

        if event == cv2.EVENT_LBUTTONUP and self._coregistration_drag_active:
            should_add_point = not self._coregistration_drag_moved
            self._coregistration_drag_active = False
            self._coregistration_drag_start_display = None
            self._coregistration_drag_start_transform = None
            moved = self._coregistration_drag_moved
            self._coregistration_drag_moved = False
            if should_add_point:
                self._handle_overlay_image_click(x, y)
            elif moved:
                self._apply_coregistration_transform()
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if hovered_action is not None:
            self._polygon_hover_action = hovered_action
            self._polygon_button_pressed_until[hovered_action] = time.monotonic() + 0.16
            self._polygon_pending_action = hovered_action
            return

        if x < POLYGON_PANEL_WIDTH:
            return

        if self._coregistration_transform is not None:
            self._coregistration_drag_active = True
            self._coregistration_drag_moved = False
            self._coregistration_drag_start_display = (x, y)
            self._coregistration_drag_start_transform = self._coregistration_transform.copy()
            return

        self._handle_overlay_image_click(x, y)

    def run_polygon_overlay(self) -> str:
        assert self._folder is not None
        assert self._metadata is not None
        points_px_full = self.build_projected_points()

        if self._enface_full is None:
            raise FileNotFoundError("Embedded DICOM enface image was not loaded correctly.")

        height, width = self._enface_full.shape[:2]
        crop_x1, crop_x2, crop_y1, crop_y2 = self._overlay_crop_bounds_px
        self._v1 = max(0, min(crop_x1, width - 1))
        self._v2 = max(self._v1 + 1, min(crop_x2, width - 1))
        self._h1 = max(0, min(crop_y1, height - 1))
        self._h2 = max(self._h1 + 1, min(crop_y2, height - 1))

        self._points_px_full = []
        self._points_px_crop = []
        for (x_full, y_full) in points_px_full:
            x_full = max(0, min(int(round(x_full)), width - 1))
            y_full = max(0, min(int(round(y_full)), height - 1))
            self._points_px_full.append((x_full, y_full))
            self._points_px_crop.append((x_full - self._v1, y_full - self._h1))

        polygon_colors = [
            (0, 210, 210),
            (0, 170, 0),
            (0, 120, 230),
            (180, 0, 180),
            (180, 80, 0),
            (0, 160, 200),
        ]
        line_colors = [
            (255, 80, 255),
            (0, 215, 255),
            (0, 190, 110),
            (255, 170, 0),
            (120, 120, 255),
            (255, 110, 110),
        ]
        polygon_shade_alpha = 0.20

        self._polygon_pending_action = None
        self._polygon_hover_action = None
        self._polygon_button_pressed_until = {}
        self.report("Enface polygon stage started")

        self.show_instructions(
            WINDOW_POLYGON_HELP,
            [
                "Choose Draw Polygon, Line Marker, Point Feature, or Delete Points",
                "Draw Polygon = Click points to define polygon",
                "Line Marker = Click sequential points for path",
                "Next Line Marker = Finish current line and start another",
                "Point Feature = Click points of interest",
                "Delete Points = Click a point feature, polygon point, or line point to remove it",
                "Fovea is marked from B-scan stage",
                "Left panel = Tools / Back / Coregister / Save",
                "u = Undo last point",
                "r = Reset current active tool marks",
                "p = Draw polygon tool",
                "n = Finish polygon and start next",
                "l = Line marker tool",
                "m = Finish line marker and start next",
                "x = Point feature tool",
                "d = Delete points tool",
                "c = Coregister uploaded image",
                "Drag mouse = Move coregistration",
                "j / k = Clockwise / anticlockwise rotate",
                "- / = = Down / up scale",
                "0 = Reset coregistration to auto",
                "b = Back to B-scans",
                "s = Save image and coordinates",
                "q = Finish session",
            ],
        )
        cv2.namedWindow(WINDOW_POLYGON, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.waitKey(1)
        cv2.setMouseCallback(WINDOW_POLYGON, self.on_polygon_click)

        while True:
            if self._should_cancel():
                self._quit_requested = True
                self._safe_destroy_window(WINDOW_POLYGON)
                self._safe_destroy_window(WINDOW_POLYGON_HELP)
                self.report("Measurement cancelled by Exit button.")
                return "quit"
            overlay = self._render_polygon_overlay_base()
            for (px, py) in self._points_px_crop:
                cv2.circle(overlay, (px + self._v1, py + self._h1), 4, (255, 0, 0), -1)

            polygon_measurements: List[PolygonMeasurement] = []
            for idx, polygon_vertices in enumerate(self.completed_polygons):
                polygon_color = polygon_colors[idx % len(polygon_colors)]
                polygon_pts = np.array(polygon_vertices, dtype=np.int32)
                shade = overlay.copy()
                cv2.fillPoly(shade, [polygon_pts], polygon_color)
                overlay = cv2.addWeighted(shade, polygon_shade_alpha, overlay, 1.0 - polygon_shade_alpha, 0)
                cv2.polylines(overlay, [polygon_pts], True, polygon_color, 2)
                for vertex_x, vertex_y in polygon_vertices:
                    vertex_point = (int(round(vertex_x)), int(round(vertex_y)))
                    cv2.circle(overlay, vertex_point, 3, polygon_color, -1, cv2.LINE_AA)
                    cv2.circle(overlay, vertex_point, 5, (255, 255, 255), 1, cv2.LINE_AA)
                polygon_measurements.append(self._polygon_metrics(polygon_vertices))

            current_color_index = len(self.completed_polygons) % len(polygon_colors)
            for vertex_x, vertex_y in self.selected_vertices:
                vertex_point = (int(round(vertex_x)), int(round(vertex_y)))
                cv2.circle(overlay, vertex_point, 5, polygon_colors[current_color_index], -1, cv2.LINE_AA)
                cv2.circle(overlay, vertex_point, 8, (255, 255, 255), 1, cv2.LINE_AA)

            if len(self.selected_vertices) >= 2:
                path_pts = np.array(self.selected_vertices, dtype=np.int32)
                cv2.polylines(overlay, [path_pts], False, polygon_colors[current_color_index], 2)

            if len(self.selected_vertices) >= 3:
                polygon_color = polygon_colors[current_color_index]
                polygon_pts = np.array(self.selected_vertices, dtype=np.int32)
                shade = overlay.copy()
                cv2.fillPoly(shade, [polygon_pts], polygon_color)
                overlay = cv2.addWeighted(shade, polygon_shade_alpha, overlay, 1.0 - polygon_shade_alpha, 0)
                cv2.polylines(overlay, [polygon_pts], True, polygon_color, 2)
                polygon_measurements.append(self._polygon_metrics(self.selected_vertices))

            all_areas = [measurement.area_mm2 for measurement in polygon_measurements]
            self._area = float(sum(all_areas)) if all_areas else None

            if polygon_measurements:
                total_polygon_area = float(sum(measurement.area_mm2 for measurement in polygon_measurements))
                cv2.putText(
                    overlay,
                    f"Total feature area = {total_polygon_area:.3f} mm^2",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 180),
                    2,
                )
                if len(polygon_measurements) >= 2:
                    label_y = 78
                    for idx, measurement in enumerate(polygon_measurements, start=1):
                        polygon_color = polygon_colors[(idx - 1) % len(polygon_colors)]
                        cv2.putText(
                            overlay,
                            f"Polygon {idx}: {measurement.area_mm2:.3f} mm^2",
                            (30, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.49,
                            polygon_color,
                            1,
                        )
                        label_y += 22

            for idx, measurement in enumerate(polygon_measurements):
                centroid_x = max(0, min(int(round(measurement.centroid_px[0])), width - 1))
                centroid_y = max(0, min(int(round(measurement.centroid_px[1])), height - 1))
                marker_colour = polygon_colors[idx % len(polygon_colors)]
                cv2.drawMarker(
                    overlay,
                    (centroid_x, centroid_y),
                    marker_colour,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=10,
                    thickness=1,
                    line_type=cv2.LINE_AA,
                )
                cv2.circle(overlay, (centroid_x, centroid_y), 3, marker_colour, 1, cv2.LINE_AA)

            if self._fovea_point is not None:
                fovea_colour = (0, 255, 255)
                cv2.drawMarker(
                    overlay,
                    self._fovea_point,
                    fovea_colour,
                    markerType=cv2.MARKER_STAR,
                    markerSize=18,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
                cv2.circle(overlay, self._fovea_point, 7, fovea_colour, 2, cv2.LINE_AA)
                cv2.putText(
                    overlay,
                    "Fovea",
                    (self._fovea_point[0] + 10, self._fovea_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    fovea_colour,
                    1,
                    cv2.LINE_AA,
                )

            rendered_line_paths = self._all_line_marker_paths(include_current=True)
            for line_idx, line_points in enumerate(rendered_line_paths, start=1):
                line_colour = line_colors[(line_idx - 1) % len(line_colors)]
                if len(line_points) >= 2:
                    line_pts = np.array(line_points, dtype=np.int32)
                    cv2.polylines(overlay, [line_pts], False, line_colour, 2, cv2.LINE_AA)
                    line_length_mm = self._line_marker_length_mm(line_points)
                    label_x, label_y = self._line_marker_label_position(line_points)
                    label_x = max(8, min(label_x, width - 170))
                    label_y = max(18, min(label_y, height - 8))
                    cv2.putText(
                        overlay,
                        f"Line {line_idx}: {line_length_mm:.2f} mm",
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        line_colour,
                        1,
                        cv2.LINE_AA,
                    )
                for point in line_points:
                    cv2.circle(overlay, point, 4, line_colour, -1, cv2.LINE_AA)
                    cv2.circle(overlay, point, 7, (255, 255, 255), 1, cv2.LINE_AA)

            point_feature_colour = (0, 165, 255)
            for feature_idx, point in enumerate(self._point_features, start=1):
                cv2.drawMarker(
                    overlay,
                    point,
                    point_feature_colour,
                    markerType=cv2.MARKER_DIAMOND,
                    markerSize=15,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
                cv2.circle(overlay, point, 5, point_feature_colour, 2, cv2.LINE_AA)
                cv2.putText(
                    overlay,
                    f"PF{feature_idx}",
                    (point[0] + 10, point[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    point_feature_colour,
                    1,
                    cv2.LINE_AA,
                )

            display_overlay = self._resize_enface_for_display(overlay)
            panel = self._build_polygon_control_panel(display_overlay.shape[0])
            cv2.imshow(WINDOW_POLYGON, np.hstack([panel, display_overlay]))
            key = cv2.waitKeyEx(30) & 0xFF
            if self._polygon_pending_action is not None:
                pending_action = self._polygon_pending_action
                self._polygon_pending_action = None
                if pending_action == "draw_polygon":
                    key = ord("p")
                elif pending_action == "next_polygon":
                    key = ord("n")
                elif pending_action == "line_marker":
                    key = ord("l")
                elif pending_action == "next_line_marker":
                    key = ord("m")
                elif pending_action == "point_feature":
                    key = ord("x")
                elif pending_action == "delete_points":
                    key = ord("d")
                elif pending_action == "back_to_bscans":
                    key = ord("b")
                elif pending_action == "save":
                    key = ord("s")
                elif pending_action == "coregister":
                    key = ord("c")

            close_requested = False
            try:
                if cv2.getWindowProperty(WINDOW_POLYGON, cv2.WND_PROP_VISIBLE) < 1:
                    close_requested = True
            except cv2.error:
                close_requested = True

            if close_requested:
                self._safe_destroy_window(WINDOW_POLYGON_HELP)
                return "finish"

            if self._coregistration_uploaded_canvas is not None and key == ord("j"):
                self._adjust_coregistration(rotation_deg=-0.5)
            elif self._coregistration_uploaded_canvas is not None and key == ord("J"):
                self._adjust_coregistration(rotation_deg=-2.0)
            elif self._coregistration_uploaded_canvas is not None and key == ord("k"):
                self._adjust_coregistration(rotation_deg=0.5)
            elif self._coregistration_uploaded_canvas is not None and key == ord("K"):
                self._adjust_coregistration(rotation_deg=2.0)
            elif self._coregistration_uploaded_canvas is not None and key == ord("-"):
                self._adjust_coregistration(scale_factor=0.99)
            elif self._coregistration_uploaded_canvas is not None and key == ord("_"):
                self._adjust_coregistration(scale_factor=0.97)
            elif self._coregistration_uploaded_canvas is not None and key == ord("="):
                self._adjust_coregistration(scale_factor=1.01)
            elif self._coregistration_uploaded_canvas is not None and key == ord("+"):
                self._adjust_coregistration(scale_factor=1.03)
            elif self._coregistration_uploaded_canvas is not None and key == ord("0"):
                self._reset_coregistration_adjustment()
            elif key == ord("p"):
                self._overlay_tool = "polygon"
                self.report("Draw Polygon tool selected.")
            elif key == ord("l"):
                self._overlay_tool = "line"
                self.report("Line Marker tool selected.")
            elif key == ord("m"):
                self._start_next_line_marker()
            elif key == ord("x"):
                self._overlay_tool = "point"
                self.report("Point Feature tool selected.")
            elif key == ord("d"):
                self._overlay_tool = "delete"
                self.report("Delete Points tool selected.")
            elif key == ord("u"):
                self._undo_active_overlay_mark()
            elif key == ord("r"):
                self._reset_active_overlay_marks()
            elif key == ord("n"):
                self._start_next_polygon()
            elif key == ord("b"):
                self._safe_destroy_window(WINDOW_POLYGON)
                self._safe_destroy_window(WINDOW_POLYGON_HELP)
                return "back"
            elif key == ord("s"):
                self._save_overlay_snapshot(overlay)
            elif key == ord("c"):
                self._coregister_uploaded_image()
            elif key == ord("q"):
                break

        self._safe_destroy_window(WINDOW_POLYGON)
        self._safe_destroy_window(WINDOW_POLYGON_HELP)
        return "finish"

    def compute_area_mm2(self, vertices_full: List[Tuple[int, int]]) -> float:
        assert self._metadata is not None
        hull_mm: List[Tuple[float, float]] = []
        for (x_full, y_full) in vertices_full:
            x_mm = float(x_full) * self._metadata.scale_x_enface
            y_mm = float(y_full) * self._metadata.scale_y_enface
            hull_mm.append((x_mm, y_mm))
        poly = Polygon(hull_mm)
        return float(poly.area)

    def _polygon_metrics(self, vertices_full: List[Tuple[int, int]]) -> PolygonMeasurement:
        assert self._metadata is not None
        hull_mm: List[Tuple[float, float]] = []
        for (x_full, y_full) in vertices_full:
            x_mm = float(x_full) * self._metadata.scale_x_enface
            y_mm = float(y_full) * self._metadata.scale_y_enface
            hull_mm.append((x_mm, y_mm))

        poly = Polygon(hull_mm)
        centroid_mm = tuple(map(float, poly.centroid.coords[0]))
        centroid_px = (
            centroid_mm[0] / self._metadata.scale_x_enface,
            centroid_mm[1] / self._metadata.scale_y_enface,
        )
        return PolygonMeasurement(
            area_mm2=float(poly.area),
            perimeter_mm=float(poly.length),
            centroid_px=centroid_px,
            centroid_mm=centroid_mm,
        )

    def compute_polygon_measurements(self, polygons_full: List[List[Tuple[int, int]]]) -> List[PolygonMeasurement]:
        return [self._polygon_metrics(vertices) for vertices in polygons_full if len(vertices) >= 3]

    def save_session_outputs(self) -> SessionResult:
        assert self._folder is not None
        assert self._metadata is not None

        all_polygon_vertices: List[Tuple[int, int]] = []
        for polygon_vertices in self.completed_polygons:
            all_polygon_vertices.extend(polygon_vertices)
        all_polygon_vertices.extend(self.selected_vertices)

        polygons_for_stats = [polygon.copy() for polygon in self.completed_polygons if len(polygon) >= 3]
        if len(self.selected_vertices) >= 3:
            polygons_for_stats.append(self.selected_vertices.copy())

        polygon_measurements = self.compute_polygon_measurements(polygons_for_stats)
        total_area = float(sum(measurement.area_mm2 for measurement in polygon_measurements)) if polygon_measurements else None
        total_perimeter = float(sum(measurement.perimeter_mm for measurement in polygon_measurements)) if polygon_measurements else None
        polygon_centroids_px = [measurement.centroid_px for measurement in polygon_measurements]
        polygon_centroids_mm = [measurement.centroid_mm for measurement in polygon_measurements]
        line_marker_paths = [line for line in self._all_line_marker_paths(include_current=True) if line]
        line_marker_length_mm = self._line_marker_length_mm()
        fovea_px = self._fovea_point
        fovea_mm: Optional[Tuple[float, float]] = None
        if fovea_px is not None:
            fovea_mm = (
                float(fovea_px[0]) * self._metadata.scale_x_enface,
                float(fovea_px[1]) * self._metadata.scale_y_enface,
            )
        point_feature_rows: List[Tuple[Tuple[int, int], Tuple[float, float]]] = []
        for point in self._point_features:
            point_feature_rows.append(
                (
                    point,
                    (
                        float(point[0]) * self._metadata.scale_x_enface,
                        float(point[1]) * self._metadata.scale_y_enface,
                    ),
                )
            )
        self._csv_output_name = "feature_area_points.csv"
        csv_path = self._folder / self._csv_output_name
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["bscan_index"]
                + [f"x_pixel_{i}" for i in range(1, MAX_FEATURE_POINTS_PER_BSCAN + 1)]
                + ["centroid_x_px", "centroid_y_px", "centroid_x_mm", "centroid_y_mm"]
            )
            for bscan_index, xvals in sorted(self.feature_coords.items()):
                row = [bscan_index] + list(xvals[:MAX_FEATURE_POINTS_PER_BSCAN])
                row += [""] * (MAX_FEATURE_POINTS_PER_BSCAN - len(row) + 1)
                row += ["", "", "", ""]
                writer.writerow(row[: MAX_FEATURE_POINTS_PER_BSCAN + 5])
            for idx, measurement in enumerate(polygon_measurements, start=1):
                writer.writerow(
                    [f"polygon_{idx}_centroid"]
                    + [""] * MAX_FEATURE_POINTS_PER_BSCAN
                    + [
                        f"{measurement.centroid_px[0]:.3f}",
                        f"{measurement.centroid_px[1]:.3f}",
                        f"{measurement.centroid_mm[0]:.6f}",
                        f"{measurement.centroid_mm[1]:.6f}",
                    ]
                )
            if fovea_px is not None and fovea_mm is not None:
                writer.writerow(
                    ["fovea_point"]
                    + [""] * MAX_FEATURE_POINTS_PER_BSCAN
                    + [
                        f"{float(fovea_px[0]):.3f}",
                        f"{float(fovea_px[1]):.3f}",
                        f"{fovea_mm[0]:.6f}",
                        f"{fovea_mm[1]:.6f}",
                    ]
                )
            for feature_idx, (feature_px, feature_mm) in enumerate(point_feature_rows, start=1):
                writer.writerow(
                    [f"point_feature_{feature_idx}"]
                    + [""] * MAX_FEATURE_POINTS_PER_BSCAN
                    + [
                        f"{float(feature_px[0]):.3f}",
                        f"{float(feature_px[1]):.3f}",
                        f"{feature_mm[0]:.6f}",
                        f"{feature_mm[1]:.6f}",
                    ]
                )
            for line_idx, line_points in enumerate(line_marker_paths, start=1):
                for point_idx, line_px in enumerate(line_points, start=1):
                    line_mm = (
                        float(line_px[0]) * self._metadata.scale_x_enface,
                        float(line_px[1]) * self._metadata.scale_y_enface,
                    )
                    writer.writerow(
                        [f"line_marker_{line_idx}_point_{point_idx}"]
                        + [""] * MAX_FEATURE_POINTS_PER_BSCAN
                        + [
                            f"{float(line_px[0]):.3f}",
                            f"{float(line_px[1]):.3f}",
                            f"{line_mm[0]:.6f}",
                            f"{line_mm[1]:.6f}",
                        ]
                    )

        measurements_file = "feature_measurements.csv"
        measurements_path = self._folder / measurements_file
        with measurements_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "value", "units"])
            writer.writerow(["polygon_count", len(polygons_for_stats), "count"])
            writer.writerow(["labelled_bscan_count", len(self.feature_coords), "count"])
            writer.writerow(["point_feature_count", len(point_feature_rows), "count"])
            writer.writerow(["line_marker_count", len(line_marker_paths), "count"])
            writer.writerow(["line_marker_point_count", sum(len(line) for line in line_marker_paths), "count"])
            writer.writerow(["line_marker_path_length", f"{line_marker_length_mm:.6f}", "mm"])
            if fovea_px is not None and fovea_mm is not None:
                writer.writerow(["fovea_x_px", f"{float(fovea_px[0]):.3f}", "px"])
                writer.writerow(["fovea_y_px", f"{float(fovea_px[1]):.3f}", "px"])
                writer.writerow(["fovea_x_mm", f"{fovea_mm[0]:.6f}", "mm"])
                writer.writerow(["fovea_y_mm", f"{fovea_mm[1]:.6f}", "mm"])
                if self._fovea_bscan_index is not None:
                    writer.writerow(["fovea_bscan_index", self._fovea_bscan_index + 1, "index"])
                if self._fovea_bscan_point is not None:
                    writer.writerow(["fovea_bscan_x_px", self._fovea_bscan_point[0], "px"])
                    writer.writerow(["fovea_bscan_y_px", self._fovea_bscan_point[1], "px"])
            for idx, measurement in enumerate(polygon_measurements, start=1):
                metric_prefix = f"polygon_{idx}"
                writer.writerow([f"{metric_prefix}_area", f"{measurement.area_mm2:.6f}", "mm^2"])
                writer.writerow([f"{metric_prefix}_perimeter", f"{measurement.perimeter_mm:.6f}", "mm"])
                writer.writerow([f"{metric_prefix}_centroid_x_px", f"{measurement.centroid_px[0]:.3f}", "px"])
                writer.writerow([f"{metric_prefix}_centroid_y_px", f"{measurement.centroid_px[1]:.3f}", "px"])
                writer.writerow([f"{metric_prefix}_centroid_x_mm", f"{measurement.centroid_mm[0]:.6f}", "mm"])
                writer.writerow([f"{metric_prefix}_centroid_y_mm", f"{measurement.centroid_mm[1]:.6f}", "mm"])
                if fovea_mm is not None:
                    centroid_to_fovea_mm = math.hypot(
                        measurement.centroid_mm[0] - fovea_mm[0],
                        measurement.centroid_mm[1] - fovea_mm[1],
                    )
                    writer.writerow([f"{metric_prefix}_centroid_to_fovea_distance", f"{centroid_to_fovea_mm:.6f}", "mm"])
            for feature_idx, (feature_px, feature_mm) in enumerate(point_feature_rows, start=1):
                writer.writerow([f"point_feature_{feature_idx}_x_px", f"{float(feature_px[0]):.3f}", "px"])
                writer.writerow([f"point_feature_{feature_idx}_y_px", f"{float(feature_px[1]):.3f}", "px"])
                writer.writerow([f"point_feature_{feature_idx}_x_mm", f"{feature_mm[0]:.6f}", "mm"])
                writer.writerow([f"point_feature_{feature_idx}_y_mm", f"{feature_mm[1]:.6f}", "mm"])
                if fovea_mm is not None:
                    feature_to_fovea_mm = math.hypot(feature_mm[0] - fovea_mm[0], feature_mm[1] - fovea_mm[1])
                    writer.writerow([f"point_feature_{feature_idx}_to_fovea_distance", f"{feature_to_fovea_mm:.6f}", "mm"])
                for polygon_idx, measurement in enumerate(polygon_measurements, start=1):
                    polygon_label = f"polygon_{polygon_idx}"
                    feature_to_centroid_mm = math.hypot(
                        feature_mm[0] - measurement.centroid_mm[0],
                        feature_mm[1] - measurement.centroid_mm[1],
                    )
                    writer.writerow(
                        [
                            f"point_feature_{feature_idx}_to_{polygon_label}_centroid_distance",
                            f"{feature_to_centroid_mm:.6f}",
                            "mm",
                        ]
                    )
            for line_idx, line_points in enumerate(line_marker_paths, start=1):
                writer.writerow([f"line_marker_{line_idx}_path_length", f"{self._line_marker_length_mm(line_points):.6f}", "mm"])
            writer.writerow(["feature_area", f"{total_area:.6f}" if total_area is not None else "", "mm^2"])
            writer.writerow(["feature_perimeter", f"{total_perimeter:.6f}" if total_perimeter is not None else "", "mm"])

        result = SessionResult(
            folder=str(self._folder),
            xml_file=self._metadata.xml_file,
            enface_file=self._metadata.enface_file,
            bscan_count=len(self._bscans),
            labelled_bscan_count=len(self.feature_coords),
            feature_coords_by_bscan=self.feature_coords,
            polygon_vertices_px=all_polygon_vertices,
            points_px_crop=self._points_px_crop,
            feature_area_mm2=total_area,
            feature_perimeter_mm=total_perimeter,
            polygon_centroids_px=polygon_centroids_px,
            polygon_centroids_mm=polygon_centroids_mm,
            overlay_image=self._overlay_output_name,
            csv_file=self._csv_output_name,
            measurements_file=measurements_file,
            overlay_opened=self._overlay_requested,
            cancelled_by_user=self._quit_requested,
            polygon_count=len(polygons_for_stats),
        )

#        output_json = self._folder / "feature_area_session.json"
#        with output_json.open("w", encoding="utf-8") as fh:
#            json.dump(asdict(result), fh, indent=2)

        self.report(f"Saved feature_area_points.csv in {self._folder}")
        self.report(f"Saved feature_measurements.csv in {self._folder}")
        return result


class OCTFeatureAreaGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.protocol("WM_DELETE_WINDOW", self.request_exit)
        self.root.geometry("1080x860")
        self.root.minsize(980, 780)

        self.folder_var = tk.StringVar()
        self.xml_var = tk.StringVar(value="Heidelberg DICOMDIR file: not selected")
        self.status_var = tk.StringVar(value="Select the DICOMDIR file in the Heidelberg OCT DICOM export folder to begin.")
        self.result_var = tk.StringVar(value="No measurement yet.")
        self.selected_xml_path: Optional[Path] = None
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._exit_requested = False

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="OCT feature mapper (Heidelberg DICOM exports)", font=("TkDefaultFont", 16, "bold"))
        title.pack(anchor="w")
        subtitle = ttk.Label(
            main,
            text="Manual B-scan feature/fovea marking with enface polygon, line, and point-feature annotation.",
        )
        subtitle.pack(anchor="w", pady=(4, 14))

        folder_frame = ttk.LabelFrame(main, text="1. Select OCT DICOM export folder via DICOMDIR", padding=12)
        folder_frame.pack(fill="x", pady=(0, 12))

        entry = ttk.Entry(folder_frame, textvariable=self.folder_var)
        entry.pack(side="left", fill="x", expand=True)
        ttk.Button(folder_frame, text="Browse DICOMDIR...", command=self.choose_xml_file).pack(side="left", padx=(8, 0))

        xml_label = ttk.Label(folder_frame, textvariable=self.xml_var)
        xml_label.pack(anchor="w", pady=(10, 0))

        actions = ttk.LabelFrame(main, text="2. Run measurement", padding=12)
        actions.pack(fill="x", pady=(0, 12))

        self.run_button = ttk.Button(actions, text="Start measurement", command=self.start_measurement)
        self.run_button.pack(side="left")
        ttk.Button(actions, text="Open selected folder", command=self.open_selected_folder).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Exit", command=self.request_exit).pack(side="right")

        instructions = ttk.LabelFrame(main, text="How to use", padding=12)
        instructions.pack(fill="x", pady=(0, 12))
        instructions_text = (
            "1. Export DICOM patient files for the right and left eyes separately. \n"
            "2. Click Browse DICOMDIR and select the DICOMDIR file inside the OCT DICOM folder.\n"
            "3. In the B-scan window, click feature points along the band (up to 6); press v then click to mark the fovea.\n"
            "4. The live enface preview sits to the right and updates as you move through B-scans.\n"
            "5. Press o at any time to open the large enface overlay.\n"
            "6. In the enface window, press b to go back to B-scans for corrections.\n"
            "7. Mark the fovea in the B-scan window with v, then use Draw Polygon, Line Marker, or Point Feature in the overlay.\n"
            "8. Use Next Polygon or Next Line Marker to start a new shape without losing the previous one.\n"
            "9. Press s to save the overlay image, then q to finish."
        )
        ttk.Label(instructions, text=instructions_text, justify="left").pack(anchor="w")

        results = ttk.LabelFrame(main, text="Session summary", padding=12)
        results.pack(fill="x", pady=(0, 12))
        ttk.Label(results, textvariable=self.result_var, justify="left").pack(anchor="w")

        log_frame = ttk.LabelFrame(main, text="Status", padding=12)
        log_frame.pack(fill="both", expand=True)
        ttk.Label(log_frame, textvariable=self.status_var, wraplength=700, justify="left").pack(anchor="w")
        self.log_box = tk.Text(log_frame, height=14, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True, pady=(10, 0))

    def append_log(self, message: str) -> None:
        try:
            if self._exit_requested and not self.root.winfo_exists():
                return
            self.status_var.set(message)
            self.log_box.configure(state="normal")
            self.log_box.insert("end", f"{message}\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        except tk.TclError:
            self._exit_requested = True

    def request_exit(self) -> None:
        self._exit_requested = True
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if self._running:
            try:
                self.append_log("Exit requested. Closing active analysis windows...")
            except tk.TclError:
                pass
            return
        self.root.destroy()

    def _pump_gui_events(self) -> None:
        try:
            if self.root.winfo_exists():
                self.root.update_idletasks()
                self.root.update()
        except tk.TclError:
            self._exit_requested = True

    def _cancel_requested(self) -> bool:
        return self._exit_requested

    def choose_xml_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select DICOMDIR file",
            filetypes=[("DICOMDIR", "DICOMDIR"), ("All files", "*.*")],
        )
        if file_path:
            self.selected_xml_path = Path(file_path)
            self.folder_var.set(str(self.selected_xml_path.parent))
            self.xml_var.set(f"DICOMDIR file: {self.selected_xml_path.name}")
            self.append_log(f"Selected DICOMDIR file: {self.selected_xml_path.name} | Folder: {self.selected_xml_path.parent}")

    def open_selected_folder(self) -> None:
        path = self.selected_xml_path.parent if self.selected_xml_path else None
        if path is None:
            messagebox.showinfo(APP_TITLE, "Please select a DICOMDIR file first.")
            return
        if not path.exists():
            messagebox.showerror(APP_TITLE, "The selected folder does not exist.")
            return
        self.append_log(f"Opening folder: {path} | DICOM file: {self.selected_xml_path.name}")
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')

    def start_measurement(self) -> None:
        if self._running:
            messagebox.showinfo(APP_TITLE, "A measurement is already running.")
            return
        if self.selected_xml_path is None:
            messagebox.showinfo(APP_TITLE, "Please select a DICOMDIR file first.")
            return
        if not self.selected_xml_path.exists() or self.selected_xml_path.name.upper() != "DICOMDIR":
            messagebox.showerror(APP_TITLE, "Please select the DICOMDIR file inside the DICOM folder.")
            return

        self._running = True
        self._exit_requested = False
        self.run_button.configure(state="disabled")
        self.result_var.set("Measurement in progress...")
        self.append_log(f"Starting measurement workflow... DICOMDIR file: {self.selected_xml_path.name}")
        self.root.after(0, lambda: self._run_workflow(self.selected_xml_path))

    def _threadsafe_log(self, message: str) -> None:
        self.root.after(0, lambda message=message: self.append_log(message))

    def _run_workflow(self, xml_path: Path) -> None:
        try:
            workflow = OCTFeatureAreaWorkflow(
                progress_callback=self.append_log,
                cancel_callback=self._cancel_requested,
                ui_callback=self._pump_gui_events,
            )
            result = workflow.run(xml_path)
            self._on_success(result)
        except Exception as exc:
            self._on_error(exc)

    def _on_success(self, result: SessionResult) -> None:
        self._running = False
        if self._exit_requested:
            self.root.destroy()
            return
        self.run_button.configure(state="normal")
        self.xml_var.set(f"DICOMDIR file: {result.xml_file}")
        if result.cancelled_by_user:
            self.result_var.set(
                "Session cancelled by user.\n"
                f"Labelled B-scans: {result.labelled_bscan_count}/{result.bscan_count}\n"
                f"DICOMDIR file: {result.xml_file}"
            )
            return
        if result.feature_area_mm2 is None:
            self.result_var.set(
                "Session finished. No final polygon area was saved.\n"
                f"Labelled B-scans: {result.labelled_bscan_count}/{result.bscan_count}\n"
                f"DICOM file: {result.xml_file}\n"
                f"Overlay opened: {'yes' if result.overlay_opened else 'no'}"
            )
        else:
            self.result_var.set(
                f"Feature area: {result.feature_area_mm2:.3f} mm^2\n"
                f"Feature perimeter: {result.feature_perimeter_mm:.3f} mm\n"
                f"Polygon centroids: {len(result.polygon_centroids_mm)} saved to CSV\n"
                f"Labelled B-scans: {result.labelled_bscan_count}/{result.bscan_count}\n"
                f"DICOM file: {result.xml_file}\n"
                f"Overlay image: {result.overlay_image or 'not saved'}\n"
                f"CSV points: {result.csv_file or 'not saved'}\n"
                f"Measurement CSV: {result.measurements_file or 'not saved'}"
            )

    def _on_error(self, exc: Exception) -> None:
        self._running = False
        if self._exit_requested:
            self.root.destroy()
            return
        self.run_button.configure(state="normal")
        self.append_log(f"Error: {exc}")
        messagebox.showerror(APP_TITLE, str(exc))

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main() -> int:
    gui = OCTFeatureAreaGUI()
    return gui.run()


if __name__ == "__main__":
    sys.exit(main())
