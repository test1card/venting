try:
    from .cli import main
except ImportError:
    from venting.cli import main

if __name__ == "__main__":
    main()
