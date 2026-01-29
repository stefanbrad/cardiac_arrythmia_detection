import wfdb
import os
from tqdm import tqdm

def download_mitbih(output_dir='data/mitbih'):
    os.makedirs(output_dir, exist_ok=True)
    
    record_numbers = [
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
        122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
        209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
        222, 223, 228, 230, 231, 232, 233, 234
    ]
    
    success_count = 0
    failed_records = []
    
    for record_num in tqdm(record_numbers, desc="Downloading records"):
        record_name = str(record_num)
        try:
            wfdb.dl_database('mitdb', output_dir, [record_name])
            success_count += 1
        except Exception as e:
            print(f"\nError downloading record {record_num}: {e}")
            failed_records.append(record_num)
    
    print(f"\nSuccessfully downloaded: {success_count}/{len(record_numbers)} records")
    if failed_records:
        print(f"Failed records: {failed_records}")


def verify_download(mitbih_dir='data/mitbih'):
    if not os.path.exists(mitbih_dir):
        print(f"Directory not found: {mitbih_dir}")
        return False
    
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
        print(f"Missing files: {missing_files}")
        return False
    
    dat_files = len([f for f in os.listdir(mitbih_dir) if f.endswith('.dat')])
    hea_files = len([f for f in os.listdir(mitbih_dir) if f.endswith('.hea')])
    atr_files = len([f for f in os.listdir(mitbih_dir) if f.endswith('.atr')])
    
    print(f"Found {dat_files} .dat files")
    print(f"Found {hea_files} .hea files")
    print(f"Found {atr_files} .atr files")
    
    if dat_files >= 40 and hea_files >= 40 and atr_files >= 40:
        print("\nDownload verified successfully!")
        return True
    else:
        print("\nDownload may be incomplete")
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
