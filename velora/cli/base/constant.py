# Copyright 2025 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from enum import StrEnum

VELORA_LOGO = """[bold {colour}]
\\ \\   / /__| | ___  _ __ __ _
 \\ \\ / / _ \\ |/ _ \\| '__/ _` |
  \\ V /  __/ | (_) | | | (_| |
   \\_/ \\___|_|\\___/|_|  \\__,_|[/bold {colour}]
"""


class Colour(StrEnum):
    """Pastel colour palette for console components."""

    SKY = "#87ceeb"
    LAVENDER = "#b4a7d6"
    AMBER = "#f0c674"
    PERIWINKLE = "#a4b9ef"
    MINT = "#98c379"
    ROSE = "#e0a3c2"
    TEAL = "#7ec8c8"
    CORAL = "#f5a97f"
    SLATE = "#b4bdc9"
    CRIMSON = "#f5a3a3"
