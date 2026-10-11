"""Agent-owned entrypoint; a spawned worker must not consume the stdin gate."""

if __name__ == "__main__":
    from switchyard_vllm_startup import main

    main()
