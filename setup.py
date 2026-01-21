"""
Setup Script for Resume Parser
Automates installation and initial configuration
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_command(command, description):
    """Run shell command with error handling"""
    print(f"⏳ {description}...")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"✅ {description} - SUCCESS\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e}\n")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print("Please upgrade Python and try again")
        return False
    
    print("✅ Python version is compatible\n")
    return True


def create_directories():
    """Create necessary directories"""
    print_header("Creating Directories")
    
    directories = ['output', 'logs', 'samples']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    print()


def install_dependencies():
    """Install Python dependencies"""
    print_header("Installing Dependencies")
    
    # Check if requirements.txt exists
    if not Path('requirements.txt').exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    success = run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    )
    
    if not success:
        return False
    
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies from requirements.txt"
    )
    
    return success


def download_spacy_model():
    """Download spaCy language model"""
    print_header("Downloading spaCy Model")
    
    success = run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Downloading en_core_web_sm model"
    )
    
    return success


def download_sentence_transformer():
    """Download sentence transformer model"""
    print_header("Downloading Sentence Transformer Model")
    
    print("⏳ Downloading sentence transformer model...")
    print("This may take a few minutes on first run...\n")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer model downloaded successfully\n")
        return True
    except Exception as e:
        print(f"❌ Failed to download model: {e}\n")
        return False


def create_sample_files():
    """Create sample files if they don't exist"""
    print_header("Creating Sample Files")
    
    # Check if sample resume exists
    sample_resume = Path('samples/sample_resume.txt')
    if not sample_resume.exists():
        print("ℹ️  Sample resume not found in samples/")
        print("Please add sample resumes to the samples/ directory\n")
    else:
        print("✅ Sample files found\n")


def verify_installation():
    """Verify installation by importing key modules"""
    print_header("Verifying Installation")
    
    modules = [
        ('spacy', 'spaCy'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('PyPDF2', 'PyPDF2'),
        ('docx', 'python-docx'),
        ('sklearn', 'scikit-learn'),
        ('streamlit', 'Streamlit')
    ]
    
    all_success = True
    
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name} imported successfully")
        except ImportError:
            print(f"❌ Failed to import {display_name}")
            all_success = False
    
    print()
    return all_success


def print_next_steps():
    """Print next steps for the user"""
    print_header("Setup Complete!")
    
    print("🎉 Resume Parser is ready to use!\n")
    print("Next Steps:")
    print("-" * 70)
    print("\n1. Test the parser:")
    print("   python main.py --resume samples/sample_resume.txt\n")
    print("2. Try job matching:")
    print("   python main.py --resume samples/sample_resume.txt --job sample_job_description.json\n")
    print("3. Launch web interface:")
    print("   streamlit run app.py\n")
    print("4. Read the documentation:")
    print("   See README.md for detailed usage instructions\n")
    print("-" * 70)
    print("\n📚 Documentation: README.md")
    print("🐛 Issues: Check logs/ directory for error logs")
    print("💡 Tips: Add your resumes to samples/ directory\n")


def main():
    """Main setup function"""
    print("\n" + "=" * 70)
    print("  🤖 AI Resume Parser - Setup Wizard")
    print("=" * 70)
    print("\nThis script will set up your Resume Parser environment.\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies")
        print("Please check your internet connection and try again")
        sys.exit(1)
    
    # Download spaCy model
    if not download_spacy_model():
        print("\n⚠️  Failed to download spaCy model")
        print("You can download it manually later with:")
        print("python -m spacy download en_core_web_sm\n")
    
    # Download sentence transformer
    if not download_sentence_transformer():
        print("\n⚠️  Failed to download sentence transformer model")
        print("The model will be downloaded automatically on first use\n")
    
    # Create sample files
    create_sample_files()
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some modules failed to import")
        print("The parser may not work correctly")
        print("Please check the error messages above\n")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)