from playwright.sync_api import sync_playwright

def capture_jira_ticket_screenshot(ticket_url: str, output_file: str = "ticket_screenshot.png"):
    """
    Captures a screenshot of a Jira ticket using Playwright.
    
    Args:
        ticket_url (str): Full URL of the Jira ticket.
        output_file (str): Path to save the screenshot.
    
    Returns:
        str: Path to the saved screenshot file.
    """

    with sync_playwright() as p:
        
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state="JIRA_ISSUE_SUMMARIZER\\jira_auth.json")
        print("=========authenticated context loaded for Playwright=========")

        # If Jira requires login, you can use context storage or login flow here
        page = context.new_page()
        print("=======created new page in Playwright=========")
        page.goto(ticket_url)
        print("=======navigated to ticket URL in Playwright=========")

        # Wait for page to load fully
        page.wait_for_load_state("load", timeout=60000)
        print("=======page loaded in Playwright=========")

        # Capture screenshot
        page.screenshot(path=output_file, full_page=True)

        browser.close()
        print(f"Screenshot saved as {output_file}")
        return output_file


if __name__ == "__main__":
    # Demo run when executed directly
    capture_jira_ticket_screenshot("https://atlassian.net/browse/issue-id")


"""This is to get the login details and save it to a file for reuse in the main agent code.
Run this once, then comment out."""
# from playwright.sync_api import sync_playwright

# with sync_playwright() as p:
#     browser = p.chromium.launch(headless=False)
#     context = browser.new_context()
#     page = context.new_page()

#     page.goto("https://atlassian.net/browse/issue-id")
#     input("Login manually then press Enter...")

#     context.storage_state(path="JIRA_ISSUE_SUMMARIZER\\jira_auth.json")
#     context = browser.new_context(storage_state="JIRA_ISSUE_SUMMARIZER\\jira_auth.json")
