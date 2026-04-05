from __future__ import annotations

import csv
import json
import os
import sys
import threading
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


DISPLAY_SCALE = 2.5
DEFAULT_INTENSITY = 100
DEFAULT_FILTER_MODE = 0
WINDOW_BSCAN = "BSCAN"
WINDOW_POLYGON = "Manual EZ Polygon"
WINDOW_BSCAN_HELP = "BSCAN HELP"
WINDOW_POLYGON_HELP = "POLYGON HELP"
APP_TITLE = "OCT feature mapper v0.89"
MAX_EZ_POINTS_PER_BSCAN = 6
HEADER_HEIGHT = 90
BOX_MARGIN_X = 20
BOX_MARGIN_Y = 14
BOX_WIDTH = 262
BOX_HEIGHT = 61
PREVIEW_PANEL_GAP = 18
PREVIEW_PANEL_WIDTH_RATIO = 0.5
PREVIEW_MAX_HEIGHT = 1000
PREVIEW_MIN_WIDTH = 260
PREVIEW_MAX_WIDTH = 1000
PREVIEW_LINE_COLOUR = (190, 190, 190)
PREVIEW_POINT_COLOUR = (255, 0, 0)
PREVIEW_CURRENT_POINT_COLOUR = (0, 0, 255)
PREVIEW_IMAGE_FILL_PANEL = False


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
    ez_coords_by_bscan: dict[int, List[int]] = field(default_factory=dict)
    polygon_vertices_px: List[Tuple[int, int]] = field(default_factory=list)
    points_px_crop: List[Tuple[int, int]] = field(default_factory=list)
    ez_area_mm2: Optional[float] = None
    overlay_image: Optional[str] = None
    csv_file: Optional[str] = None
    overlay_opened: bool = False
    cancelled_by_user: bool = False
    polygon_count: int = 0


class OCTEZAreaWorkflow:
    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        self.progress_callback = progress_callback or (lambda message: None)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.clahe_on = False
        self.selected_vertices: List[Tuple[int, int]] = []
        self.completed_polygons: List[List[Tuple[int, int]]] = []
        self.current_points: List[Tuple[int, int]] = []
        self.ez_coords: dict[int, List[int]] = {}
        self.ez_click_points_by_bscan: dict[int, List[Tuple[int, int]]] = {}
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
        self._pending_bscan_step = 0

    def report(self, message: str) -> None:
        self.progress_callback(message)

    def show_save_confirmation(self, message: str) -> None:
        def _show():
            messagebox.showinfo(APP_TITLE, message)
    
        try:
            root = tk._default_root
            if root:
                root.after(0, _show)
        except Exception:
            pass

    def run(self, xml_path: Path) -> SessionResult:
        try:
            self._xml_path = xml_path
            self._folder = xml_path.parent
            self._overlay_requested = False
            self._quit_requested = False
            self._last_bscan_index = 0
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
                        if not self.ez_coords:
                            self.report("Cannot open the enface overlay until at least one B-scan has been labelled.")
                            stage = "bscan"
                        else:
                            self._overlay_requested = True
                            stage = "polygon"
                        continue
                    stage = "bscan"

                if stage == "polygon":
                    action = self.run_polygon_overlay()
                    if action == "back":
                        self.report("Returned from enface overlay to B-scan editing.")
                        stage = "bscan"
                        continue
                    break

            result = self.save_session_outputs()
            if result.ez_area_mm2 is not None:
                self.report(f"Finished. EZ area = {result.ez_area_mm2:.3f} mm^2")
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
        cv2.imshow(title, img)
        cv2.moveWindow(title, 1060, 30)


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

        if len(self.current_points) >= MAX_EZ_POINTS_PER_BSCAN:
            return
        self.current_points.append((x_orig, y_orig))

    def _store_current_points_for_bscan(self, idx: int) -> bool:
        if len(self.current_points) >= 2:
            saved_points = [tuple(point) for point in self.current_points[:MAX_EZ_POINTS_PER_BSCAN]]
            self.ez_click_points_by_bscan[idx + 1] = saved_points
            self.ez_coords[idx + 1] = sorted([point[0] for point in saved_points])
            self.report(f"Saved {len(saved_points)} EZ marks for B-scan {idx + 1}")
            return True
        return False

    def _commit_current_bscan_before_overlay(self, idx: int) -> None:
        stored = self._store_current_points_for_bscan(idx)
        if not stored and (idx + 1) in self.ez_click_points_by_bscan:
            saved_points = [tuple(point) for point in self.ez_click_points_by_bscan[idx + 1][:MAX_EZ_POINTS_PER_BSCAN]]
            self.current_points = saved_points
            self.ez_coords[idx + 1] = sorted([point[0] for point in saved_points])

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
                "Left click = Mark EZ points (up to 6)",
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
            self.current_points = [tuple(point) for point in self.ez_click_points_by_bscan.get(idx + 1, [])]
            self.report(f"B-scan {idx + 1} of {total}")

            while True:
                display = self.prepare_bscan_display(self._current_bscan_original)
                cv2.imshow(WINDOW_BSCAN, display)
                key = cv2.waitKey(20) & 0xFF

                if self._pending_bscan_step != 0:
                    if len(self.current_points) >= 2:
                        self._store_current_points_for_bscan(idx)
                    idx = (idx + self._pending_bscan_step) % total
                    self._pending_bscan_step = 0
                    break

                if key == ord("b"):
                    if len(self.current_points) >= 2:
                        self._store_current_points_for_bscan(idx)
                    idx = (idx - 1) % total
                    break
                elif key == ord("u"):
                    self.current_points = []
                elif key == ord("f"):
                    if len(self.current_points) >= 2:
                        self._store_current_points_for_bscan(idx)
                    idx = (idx + 1) % total
                    break
                elif key == ord("o"):
                    self._commit_current_bscan_before_overlay(idx)
                    self._last_bscan_index = idx
                    cv2.destroyWindow(WINDOW_BSCAN)
                    cv2.destroyWindow(WINDOW_BSCAN_HELP)
                    self.report("Opening enface overlay from the current measurement state.")
                    return "overlay"
                elif key == ord("q"):
                    self._last_bscan_index = idx
                    cv2.destroyWindow(WINDOW_BSCAN)
                    cv2.destroyWindow(WINDOW_BSCAN_HELP)
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

    def _collect_preview_points(self) -> List[Tuple[int, int, bool]]:
        preview_points: List[Tuple[int, int, bool]] = []
        for idx_key, xvals in sorted(self.ez_coords.items()):
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

        for x_map, y_map, is_current in self._collect_preview_points():
            x_map = max(0, min(int(round(x_map)), w - 1))
            y_map = max(0, min(int(round(y_map)), h - 1))
            colour = PREVIEW_CURRENT_POINT_COLOUR if is_current else PREVIEW_POINT_COLOUR
            radius = 4 if is_current else 3
            cv2.circle(preview, (x_map, y_map), radius, colour, -1, cv2.LINE_AA)

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

        for pt in self.current_points:
            x_disp = int(pt[0] * DISPLAY_SCALE)
            y_disp = int(pt[1] * DISPLAY_SCALE) + self._current_display_y_offset
            cv2.circle(display, (x_disp, y_disp), 6, (0, 0, 255), -1)

        preview_panel = self._build_live_enface_preview(display.shape[0], display.shape[1])
        gap = np.zeros((display.shape[0], PREVIEW_PANEL_GAP, 3), dtype=np.uint8)
        combined = np.hstack([display, gap, preview_panel])
        return combined

    def build_projected_points(self) -> List[Tuple[int, int]]:
        points_px: List[Tuple[int, int]] = []
        for idx_key, xvals in sorted(self.ez_coords.items()):
            ref_index = idx_key - 1
            if ref_index < 0 or ref_index >= len(self._frame_reference_coords_px):
                continue
            ref_coords = self._frame_reference_coords_px[ref_index]
            for xpix in xvals:
                points_px.append(self._project_bscan_x_to_overlay(ref_coords, xpix))
        return points_px

    def on_polygon_click(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or not self._points_px_crop:
            return
        x_crop = x - self._v1
        y_crop = y - self._h1
        distances = [np.hypot(x_crop - px, y_crop - py) for (px, py) in self._points_px_crop]
        nearest_index = int(np.argmin(distances))
        x_full = self._points_px_crop[nearest_index][0] + self._v1
        y_full = self._points_px_crop[nearest_index][1] + self._h1
        self.selected_vertices.append((x_full, y_full))

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

        self._points_px_crop = []
        for (x_full, y_full) in points_px_full:
            x_full = max(0, min(int(round(x_full)), width - 1))
            y_full = max(0, min(int(round(y_full)), height - 1))
            self._points_px_crop.append((x_full - self._v1, y_full - self._h1))

        polygon_colors = [
            (0, 210, 210),
            (0, 170, 0),
            (0, 120, 230),
            (180, 0, 180),
            (180, 80, 0),
            (0, 160, 200),
        ]
        polygon_shade_alpha = 0.20

        self.selected_vertices = []
        self.completed_polygons = []
        self._area = None
        self.report("Enface polygon stage started")

        self.show_instructions(
            WINDOW_POLYGON_HELP,
            [
                "Click points to define polygon",
                "u = Undo last point",
                "r = Reset current polygon",
                "n = Finish polygon and start next",
                "b = Back to B-scans",
                "s = Save image and coordinates",
                "q = Finish session",
            ],
        )
        cv2.namedWindow(WINDOW_POLYGON, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_POLYGON, self.on_polygon_click)

        while True:
            overlay = self._enface_full.copy()
            for (px, py) in self._points_px_crop:
                cv2.circle(overlay, (px + self._v1, py + self._h1), 4, (255, 0, 0), -1)

            displayed_areas: List[float] = []
            for idx, polygon_vertices in enumerate(self.completed_polygons):
                polygon_color = polygon_colors[idx % len(polygon_colors)]
                polygon_pts = np.array(polygon_vertices, dtype=np.int32)
                shade = overlay.copy()
                cv2.fillPoly(shade, [polygon_pts], polygon_color)
                overlay = cv2.addWeighted(shade, polygon_shade_alpha, overlay, 1.0 - polygon_shade_alpha, 0)
                cv2.polylines(overlay, [polygon_pts], True, polygon_color, 2)
                displayed_areas.append(self.compute_area_mm2(polygon_vertices))

            current_polygon_area: Optional[float] = None
            current_color_index = len(self.completed_polygons) % len(polygon_colors)
            if len(self.selected_vertices) >= 2:
                path_pts = np.array(self.selected_vertices, dtype=np.int32)
                cv2.polylines(overlay, [path_pts], False, polygon_colors[current_color_index][1], 2)

            if len(self.selected_vertices) >= 3:
                polygon_color = polygon_colors[current_color_index]
                polygon_pts = np.array(self.selected_vertices, dtype=np.int32)
                shade = overlay.copy()
                cv2.fillPoly(shade, [polygon_pts], polygon_color)
                overlay = cv2.addWeighted(shade, polygon_shade_alpha, overlay, 1.0 - polygon_shade_alpha, 0)
                cv2.polylines(overlay, [polygon_pts], True, polygon_color, 2)
                current_polygon_area = self.compute_area_mm2(self.selected_vertices)

            all_areas = displayed_areas.copy()
            if current_polygon_area is not None:
                all_areas.append(current_polygon_area)
            self._area = float(sum(all_areas)) if all_areas else None

            if self._area is not None:
                cv2.putText(
                    overlay,
                    f"Total EZ Area = {self._area:.3f} mm^2",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 180),
                    2,
                )
                if len(all_areas) >= 2:
                    label_y = 78
                    for idx, area in enumerate(all_areas, start=1):
                        polygon_color = polygon_colors[(idx - 1) % len(polygon_colors)]
                        cv2.putText(
                            overlay,
                            f"Polygon {idx}: {area:.3f} mm^2",
                            (30, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.49,
                            polygon_color,
                            1,
                        )
                        label_y += 22

            cv2.imshow(WINDOW_POLYGON, overlay)
            key = cv2.waitKey(30) & 0xFF

            close_requested = False
            try:
                if cv2.getWindowProperty(WINDOW_POLYGON, cv2.WND_PROP_VISIBLE) < 1:
                    close_requested = True
            except cv2.error:
                close_requested = True

            if close_requested:
                if self._area is not None:
                    self._overlay_output_name = f"Manual_EZ_Area_{self._area:.3f}mm2.png"
                    cv2.imwrite(str(self._folder / self._overlay_output_name), overlay)
                    self.report(f"Saved overlay image: {self._overlay_output_name}")
                cv2.destroyWindow(WINDOW_POLYGON_HELP)
                return "finish"

            if key == ord("u") and self.selected_vertices:
                self.selected_vertices.pop()
            elif key == ord("r"):
                self.selected_vertices = []
            elif key == ord("n") and len(self.selected_vertices) >= 3:
                self.completed_polygons.append(self.selected_vertices.copy())
                self.selected_vertices = []
                self.report(f"Started next polygon. Completed polygons: {len(self.completed_polygons)}")
            elif key == ord("b"):
                self.selected_vertices = []
                self.completed_polygons = []
                self._area = None
                cv2.destroyWindow(WINDOW_POLYGON)
                cv2.destroyWindow(WINDOW_POLYGON_HELP)
                return "back"
            elif key == ord("s") and self._area is not None:
                self._overlay_output_name = f"Manual_EZ_Area_{self._area:.3f}mm2.png"
                cv2.imwrite(str(self._folder / self._overlay_output_name), overlay)
                self.report(f"Saved overlay image: {self._overlay_output_name}")
            
                cv2.destroyWindow(WINDOW_POLYGON)
                cv2.destroyWindow(WINDOW_POLYGON_HELP)
            
                messagebox.showinfo(
                    APP_TITLE,
                    f"Saved overlay image and coordinate files in\n{self._folder}\n\nXML file:\n{self._metadata.xml_file}"
                )
            
                cv2.namedWindow(WINDOW_POLYGON, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(WINDOW_POLYGON, self.on_polygon_click)
                self.show_instructions(
                    WINDOW_POLYGON_HELP,
                    [
                        "Click points to define polygon",
                        "u = Undo last point",
                        "r = Reset current polygon",
                        "n = Finish polygon and start next",
                        "b = Back to B-scans",
                        "s = Save image and coordinates",
                        "q = Finish session",
                    ],
                
                )
            elif key == ord("q"):
                if self._area is not None:
                    self._overlay_output_name = f"Manual_EZ_Area_{self._area:.3f}mm2.png"
                    cv2.imwrite(str(self._folder / self._overlay_output_name), overlay)
                    self.report(f"Saved overlay image: {self._overlay_output_name}")
                break

        cv2.destroyWindow(WINDOW_POLYGON)
        cv2.destroyWindow(WINDOW_POLYGON_HELP)
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

    def save_session_outputs(self) -> SessionResult:
        assert self._folder is not None
        assert self._metadata is not None

        self._csv_output_name = "ez_area_points.csv"
        csv_path = self._folder / self._csv_output_name
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["bscan_index"] + [f"x_pixel_{i}" for i in range(1, MAX_EZ_POINTS_PER_BSCAN + 1)])
            for bscan_index, xvals in sorted(self.ez_coords.items()):
                row = [bscan_index] + list(xvals[:MAX_EZ_POINTS_PER_BSCAN])
                row += [""] * (MAX_EZ_POINTS_PER_BSCAN - len(row) + 1)
                writer.writerow(row[: MAX_EZ_POINTS_PER_BSCAN + 1])

        all_polygon_vertices: List[Tuple[int, int]] = []
        for polygon_vertices in self.completed_polygons:
            all_polygon_vertices.extend(polygon_vertices)
        all_polygon_vertices.extend(self.selected_vertices)

        result = SessionResult(
            folder=str(self._folder),
            xml_file=self._metadata.xml_file,
            enface_file=self._metadata.enface_file,
            bscan_count=len(self._bscans),
            labelled_bscan_count=len(self.ez_coords),
            ez_coords_by_bscan=self.ez_coords,
            polygon_vertices_px=all_polygon_vertices,
            points_px_crop=self._points_px_crop,
            ez_area_mm2=self._area,
            overlay_image=self._overlay_output_name,
            csv_file=self._csv_output_name,
            overlay_opened=self._overlay_requested,
            cancelled_by_user=self._quit_requested,
            polygon_count=len(self.completed_polygons) + (1 if len(self.selected_vertices) >= 3 else 0),
        )

#        output_json = self._folder / "ez_area_session.json"
#        with output_json.open("w", encoding="utf-8") as fh:
#            json.dump(asdict(result), fh, indent=2)

        self.report(f"Saved ez_area_points.csv in {self._folder}")
        return result


class OCTEZAreaGUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("780x620")
        self.root.minsize(740, 580)

        self.folder_var = tk.StringVar()
        self.xml_var = tk.StringVar(value="Heidelberg DICOMDIR file: not selected")
        self.status_var = tk.StringVar(value="Select the DICOMDIR file in the Heidelberg OCT DICOM export folder to begin.")
        self.result_var = tk.StringVar(value="No measurement yet.")
        self.selected_xml_path: Optional[Path] = None
        self._worker: Optional[threading.Thread] = None
        self._running = False

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="OCT feature mapper (Heidelberg DICOM exports)", font=("TkDefaultFont", 16, "bold"))
        title.pack(anchor="w")
        subtitle = ttk.Label(
            main,
            text="Manual B-scan EZ marking with transfer to enface polygon measurement.",
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
        ttk.Button(actions, text="Exit", command=self.root.destroy).pack(side="right")

        instructions = ttk.LabelFrame(main, text="How to use", padding=12)
        instructions.pack(fill="x", pady=(0, 12))
        instructions_text = (
            "1. Export DICOM patient files for the right and left eyes separately. \n"
            "2. Click Browse DICOMDIR and select the DICOMDIR file inside the OCT DICOM folder.\n"
            "3. In the B-scan window, click EZ points along the band (up to 6), then press f, b, or o to register them.\n"
            "4. The live enface preview sits to the right and updates as you move through B-scans.\n"
            "5. Press o when you want to open the large enface overlay.\n"
            "6. In the enface window, press b to go back to B-scans for corrections.\n"
            "7. Press o again to reopen a clean enface with overlaid points and redraw the polygon.\n"
            "8. Press s to save the overlay image, then q to finish."
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
        self.status_var.set(message)
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

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
        self.run_button.configure(state="disabled")
        self.result_var.set("Measurement in progress...")
        self.append_log(f"Starting measurement workflow... DICOMDIR file: {self.selected_xml_path.name}")
        self.root.after(0, lambda: self._run_workflow(self.selected_xml_path))

    def _threadsafe_log(self, message: str) -> None:
        self.root.after(0, lambda message=message: self.append_log(message))

    def _run_workflow(self, xml_path: Path) -> None:
        try:
            workflow = OCTEZAreaWorkflow(progress_callback=self.append_log)
            result = workflow.run(xml_path)
            self._on_success(result)
        except Exception as exc:
            self._on_error(exc)

    def _on_success(self, result: SessionResult) -> None:
        self._running = False
        self.run_button.configure(state="normal")
        self.xml_var.set(f"DICOMDIR file: {result.xml_file}")
        if result.cancelled_by_user:
            self.result_var.set(
                "Session cancelled by user.\n"
                f"Labelled B-scans: {result.labelled_bscan_count}/{result.bscan_count}\n"
                f"DICOMDIR file: {result.xml_file}"
            )
            messagebox.showinfo(APP_TITLE, "Measurement cancelled and returned to the main GUI.")
            return
        if result.ez_area_mm2 is None:
            self.result_var.set(
                "Session finished. No final polygon area was saved.\n"
                f"Labelled B-scans: {result.labelled_bscan_count}/{result.bscan_count}\n"
                f"DICOM file: {result.xml_file}\n"
                f"Overlay opened: {'yes' if result.overlay_opened else 'no'}"
            )
        else:
            self.result_var.set(
                f"EZ area: {result.ez_area_mm2:.3f} mm^2\n"
                f"Labelled B-scans: {result.labelled_bscan_count}/{result.bscan_count}\n"
                f"DICOM file: {result.xml_file}\n"
                f"Overlay image: {result.overlay_image or 'not saved'}\n"
                f"CSV points: {result.csv_file or 'not saved'}"
            )
        messagebox.showinfo(APP_TITLE, "Measurement finished. Results were saved in the selected folder.")

    def _on_error(self, exc: Exception) -> None:
        self._running = False
        self.run_button.configure(state="normal")
        self.append_log(f"Error: {exc}")
        messagebox.showerror(APP_TITLE, str(exc))

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main() -> int:
    gui = OCTEZAreaGUI()
    return gui.run()


if __name__ == "__main__":
    sys.exit(main())
