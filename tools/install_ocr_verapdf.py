#!/usr/bin/env python3
"""Install the pinned independent PDF/A validator for hosted export CI."""

import argparse
import hashlib
from io import BytesIO
from pathlib import Path
import subprocess
import urllib.request
from xml.sax.saxutils import escape
from zipfile import ZipFile


VERSION = "1.30.2"
URL = "https://software.verapdf.org/releases/1.30/verapdf-greenfield-1.30.2-installer.zip"
SHA256 = "6cc6341cb1af644044054b81f00a6590a7918abb18f762243de115258bcad838"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    options = parser.parse_args()
    prefix = options.prefix.resolve()
    staging = prefix.parent / "verapdf-installer"
    staging.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=90) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != SHA256:
        raise RuntimeError("Pinned veraPDF installer checksum mismatch")
    with ZipFile(BytesIO(archive)) as package:
        jar = f"verapdf-greenfield-{VERSION}/verapdf-izpack-installer-{VERSION}.jar"
        installer = staging / "installer.jar"
        installer.write_bytes(package.read(jar))
    configuration = staging / "install.xml"
    configuration.write_text(f'''<AutomatedInstallation langpack="eng">
  <com.izforge.izpack.panels.htmlhello.HTMLHelloPanel id="welcome"/>
  <com.izforge.izpack.panels.target.TargetPanel id="install_dir">
    <installpath>{escape(str(prefix))}</installpath>
  </com.izforge.izpack.panels.target.TargetPanel>
  <com.izforge.izpack.panels.packs.PacksPanel id="sdk_pack_select">
    <pack index="0" name="veraPDF GUI" selected="false"/>
    <pack index="1" name="veraPDF CLI" selected="true"/>
    <pack index="2" name="veraPDF Documentation" selected="false"/>
    <pack index="3" name="veraPDF Sample Plugins" selected="false"/>
  </com.izforge.izpack.panels.packs.PacksPanel>
  <com.izforge.izpack.panels.install.InstallPanel id="install"/>
  <com.izforge.izpack.panels.finish.FinishPanel id="finish"/>
</AutomatedInstallation>''')
    subprocess.run(["java", "-jar", str(installer), str(configuration)], check=True, timeout=120)
    subprocess.run([str(prefix / "verapdf"), "--version"], check=True, timeout=30)


if __name__ == "__main__":
    main()
