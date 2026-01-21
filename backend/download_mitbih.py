"""
Download MIT-BIH Arrhythmia Database

This script automatically downloads the MIT-BIH database from PhysioNet
"""

import wfdb
import os
from tqdm import tqdm

def download_mitbih(output_dir='data/mitbih'):
    """
    Download MIT-BIH Arrhythmia Database
    
    Args:
        output_dir: Directory to save the database
    """
    print("=" * 60)
    print("📥 Downloading MIT-BIH Arrhythmia Database")
    print("=" * 60)
    print("\nDatabase Info:")
    print("  - 48 half-hour ECG recordings")
    print("  - 47 subjects (25 men, 22 women)")
    print("  - Sampling rate: 360 Hz")
    print("  - Various arrhythmia annotations")
    print("  - Size: ~110 MB")
    print("\nSource: https://physionet.org/content/mitdb/1.0.0/")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # All MIT-BIH record numbers
    record_numbers = [
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
        122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
        209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
        222, 223, 228, 230, 231, 232, 233, 234
    ]
    
    print(f"\nDownloading {len(record_numbers)} records to {output_dir}...")
    print("This may take several minutes...\n")
    
    success_count = 0
    failed_records = []
    
    for record_num in tqdm(record_numbers, desc="Downloading records"):
        record_name = str(record_num)
        
        try:
            # Download record and annotation
            wfdb.dl_database('mitdb', output_dir, [record_name])
            success_count += 1
        except Exception as e:
            print(f"\nError downloading record {record_num}: {e}")
            failed_records.append(record_num)
    
    print("\n" + "=" * 60)
    print("✓ Download complete!")
    print("=" * 60)
    print(f"Successfully downloaded: {success_count}/{len(record_numbers)} records")
    
    if failed_records:
        print(f"Failed records: {failed_records}")
    
    print(f"\nDatabase location: {os.path.abspath(output_dir)}")
    print("\nNext steps:")
    print(f"  python train_mitbih.py --mitbih-path {output_dir}")
    print("=" * 60)


def verify_download(mitbih_dir='data/mitbih'):
    """
    Verify that MIT-BIH database was downloaded correctly
    
    Args:
        mitbih_dir: Directory containing MIT-BIH database
    """
    print("\n🔍 Verifying download...")
    
    if not os.path.exists(mitbih_dir):
        print(f"❌ Directory not found: {mitbih_dir}")
        return False
    
    # Check for some key files
    test_records = [100, 101, 200, 201]
    missing_files = []
    
    for record_num in test_records:
        dat_file = os.path.join(mitbih_dir, f"{record_num}.dat")
        hea_file = os.path.join(mitbih_dir, f"{record_num}.hea")
        atr_file = os.path.join(mitbih_dir, f"{record_num}.atr")
        
        if not os.path.exists(dat_file):
            missing_files.append(f"{record_num}.dat")
        if not os.path.exists(hea_file):
            missing_files.append(f"{record_num}.hea")
        if not os.path.exists(atr_file):
            missing_files.append(f"{record_num}.atr")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    # Count total files
    dat_files = len([f for f in os.listdir(mitbih_dir) if f.endswith('.dat')])
    hea_files = len([f for f in os.listdir(mitbih_dir) if f.endswith('.hea')])
    atr_files = len([f for f in os.listdir(mitbih_dir) if f.endswith('.atr')])
    
    print(f"✓ Found {dat_files} .dat files")
    print(f"✓ Found {hea_files} .hea files")
    print(f"✓ Found {atr_files} .atr files")
    
    if dat_files >= 40 and hea_files >= 40 and atr_files >= 40:
        print("\n✓ Download verified successfully!")
        return True
    else:
        print("\n⚠ Download may be incomplete")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download MIT-BIH Arrhythmia Database')
    parser.add_argument('--output-dir', type=str, default='data/mitbih',
                       help='Directory to save the database')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing download')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_download(args.output_dir)
    else:
        download_mitbih(args.output_dir)
        verify_download(args.output_dir)
