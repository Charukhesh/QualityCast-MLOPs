import os

def test_dataset_structure():
    """Verify data exists where expected (DVC check)"""
    train_path = "data/raw/"
    assert os.path.exists(train_path)
    # Check if we have both classes
    classes = os.listdir(train_path)
    assert "ok_front" in classes
    assert "def_front" in classes