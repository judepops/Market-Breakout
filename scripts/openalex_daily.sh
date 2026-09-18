#!/bin/bash
# RETIRED 2026-07-15. The OpenAlex citation-enrichment thesis was tested on the
# ~45k papers already fetched and found redundant with the free "team experience"
# feature (AUPRC delta +0.002, p=0.49). We are NOT completing the fetch.
#
# This script is a deliberate no-op so any leftover crontab entry does nothing.
# To remove the cron entry entirely: `crontab -e` and delete the openalex_daily line.
exit 0
