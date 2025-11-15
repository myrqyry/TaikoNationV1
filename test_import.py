import sys

print("--- Testing TaikoNation Package Imports ---")
print("Python Executable:", sys.executable)
print("Python Version:", sys.version)
print("Current Working Directory:", sys.path[0])
print("Full sys.path:", sys.path)

try:
    print("\nAttempting to import MediaPipeAudioAnalyzer...")
    from taikonation.data.mediapipe_audio import MediaPipeAudioAnalyzer
    print("SUCCESS: MediaPipeAudioAnalyzer imported successfully.")

    analyzer = MediaPipeAudioAnalyzer()
    print("SUCCESS: MediaPipeAudioAnalyzer instantiated.")

except ImportError as e:
    print(f"\nERROR: Failed to import MediaPipeAudioAnalyzer.")
    print(f"ImportError: {e}")
    # In case of an import error, let's see if we can at least find the parent package
    try:
        import taikonation
        print("NOTE: The 'taikonation' package IS found at:", taikonation.__path__)
        import taikonation.data
        print("NOTE: The 'taikonation.data' subpackage IS found at:", taikonation.data.__path__)
    except ImportError:
        print("CRITICAL: The base 'taikonation' package itself could not be found.")

except Exception as e:
    print(f"\nERROR: An unexpected error occurred: {e}")

print("\n--- Test Complete ---")
