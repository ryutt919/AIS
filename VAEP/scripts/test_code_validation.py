"""
코드 검증 스크립트 - 의존성 없이 코드 구조만 확인
"""

import ast
import os

def validate_python_file(filepath):
    """Python 파일의 구문 오류 확인"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def check_imports(filepath):
    """필요한 import 확인"""
    required_imports = {
        'preprocess_wyscout_atomic.py': [
            'socceraction.spadl',
            'socceraction.atomic.spadl',
            'socceraction.data.wyscout',
            'pandas',
            'tqdm'
        ],
        'train_vaep_model_atomic.py': [
            'socceraction.atomic.spadl',
            'socceraction.atomic.vaep.features',
            'torch',
            'pandas',
            'numpy'
        ]
    }
    
    filename = os.path.basename(filepath)
    if filename not in required_imports:
        return True, []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        missing = []
        for imp in required_imports[filename]:
            # 간단한 문자열 검색
            if imp.replace('.', ' ') not in code.replace('.', ' '):
                missing.append(imp)
        
        return len(missing) == 0, missing
    except Exception as e:
        return False, [str(e)]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    files_to_check = [
        'preprocess_wyscout_atomic.py',
        'train_vaep_model_atomic.py',
        'test_atomic_spadl_quick.py'
    ]
    
    print("=" * 80)
    print("코드 검증")
    print("=" * 80)
    
    all_ok = True
    
    for filename in files_to_check:
        filepath = os.path.join(script_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"\n✗ {filename}: 파일이 존재하지 않습니다")
            all_ok = False
            continue
        
        print(f"\n{filename}:")
        
        # 구문 검사
        is_valid, error = validate_python_file(filepath)
        if is_valid:
            print("  ✓ 구문 오류 없음")
        else:
            print(f"  ✗ 구문 오류: {error}")
            all_ok = False
        
        # Import 확인
        if filename in ['preprocess_wyscout_atomic.py', 'train_vaep_model_atomic.py']:
            imports_ok, missing = check_imports(filepath)
            if imports_ok:
                print("  ✓ 필요한 import 확인")
            else:
                print(f"  ⚠ 누락된 import: {missing}")
    
    print("\n" + "=" * 80)
    if all_ok:
        print("✓ 모든 파일 검증 완료")
    else:
        print("✗ 일부 파일에 문제가 있습니다")
    print("=" * 80)
    
    return all_ok

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

