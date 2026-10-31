"""Legacy ticket-data migration is intentionally disabled.

The current schema creates TicketRow as the only ticket table and does not
retain business-ticket history from the legacy database. Runtime schema reset
is handled by ``initialize_ticket_schema``.
"""


def main() -> None:
    raise SystemExit(
        "Legacy ticket migration is disabled; historical ticket data is not retained. "
        "Start the application to initialize the TicketRow schema."
    )


if __name__ == "__main__":
    main()
