class TestSuite:
    """Test suite for MCP server functionality"""
    
    def __init__(self, server_path: str):
        self.server_path = server_path
        self.temp_dir = None
        self.test_results = {}
        
    async def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 60)
        print("MCP Content & Media Processing Server Test Suite")
        print("=" * 60)
        
        # Setup test environment
        await self.setup_test_environment()
        
        try:
            # Connect once for all tests
            async with MCPContentClient(self.server_path) as client:
                self.test_results["server_connection"] = await self.test_server_connection(client)
                self.test_results["tools_listing"] = await self.test_tools_listing(client)
                self.test_results["system_status"] = await self.test_system_status(client)
                self.test_results["document_processing"] = await self.test_document_processing(client)
                self.test_results["image_processing"] = await self.test_image_processing(client)
                self.test_results["news_functionality"] = await self.test_news_functionality(client)
                self.test_results["resource_management"] = await self.test_resource_management(client)
                
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
        finally:
            # Cleanup
            await self.cleanup_test_environment()
            self.print_summary()

    async def setup_test_environment(self):
        """Setup test environment"""
        print("\n📁 Setting up test environment...")
        self.temp_dir = Path(tempfile.mkdtemp(prefix="mcp_test_"))
        logger.info(f"Created temp directory: {self.temp_dir}")
        await self.create_test_files()
        logger.info("Created test files")

    async def create_test_files(self):
        """Create test files for processing"""
        test_doc = self.temp_dir / "test_document.txt"
        with open(test_doc, 'w', encoding='utf-8') as f:
            f.write("# Test Document\n\nThis is a test document for MCP processing.\n\n")
            f.write("It contains multiple paragraphs and formatting.\n\n")
            f.write("- List item 1\n- List item 2\n- List item 3\n")
        
        test_md = self.temp_dir / "test_document.md"
        with open(test_md, 'w', encoding='utf-8') as f:
            f.write("# Test Markdown\n\nThis is **bold** and *italic* text.\n\n")
            f.write("```python\nprint('Hello World')\n```\n")
        
        try:
            from PIL import Image
            test_img = self.temp_dir / "test_image.png"
            img = Image.new('RGB', (200, 200), color='blue')
            img.save(test_img)
            logger.info("Created test image")
        except Exception as e:
            logger.warning(f"Could not create test image: {e}")
        
        test_email = self.temp_dir / "test.eml"
        with open(test_email, 'w', encoding='utf-8') as f:
            f.write("From: test@example.com\n")
            f.write("To: recipient@example.com\n")
            f.write("Subject: Test Email\n")
            f.write("Date: Mon, 1 Jan 2024 12:00:00 +0000\n\n")
            f.write("This is a test email message.\n")
        
    async def test_server_connection(self, client):
        """Test basic server connection"""
        print("\n🔗 Testing server connection...")
        try:
            tools = await client.list_tools()
            if tools:
                self.test_results["server_connection"] = {"status": "PASS", "message": f"Connected successfully, found {len(tools)} tools"}
                print(f"   ✅ Connection successful ({len(tools)} tools available)")
            else:
                self.test_results["server_connection"] = {"status": "FAIL", "message": "No tools available"}
                print("   ❌ Connection failed - no tools available")
        except Exception as e:
            self.test_results["server_connection"] = {"status": "FAIL", "message": str(e)}
            print(f"   ❌ Connection failed: {e}")
        return self.test_results["server_connection"]["status"] == "PASS"

    async def test_tools_listing(self, client):
        """Test tools listing functionality"""
        print("\n🛠️  Testing tools listing...")
        try:
            tools = await client.list_tools()
            expected_tools = ["convert_document", "process_image", "search_emails", "analyze_communication_patterns", "add_news_source", "get_trending_topics", "list_supported_formats"]
            found_tools = [tool.name for tool in tools]
            missing_tools = [tool for tool in expected_tools if tool not in found_tools]
            if not missing_tools:
                self.test_results["tools_listing"] = {"status": "PASS", "message": f"All {len(expected_tools)} expected tools found"}
                print(f"   ✅ All expected tools found ({len(tools)} total)")
                for tool in tools:
                    print(f"      - {tool.name}")
            else:
                self.test_results["tools_listing"] = {"status": "PARTIAL", "message": f"Missing tools: {missing_tools}"}
                print(f"   ⚠️  Missing tools: {missing_tools}")
        except Exception as e:
            self.test_results["tools_listing"] = {"status": "FAIL", "message": str(e)}
            print(f"   ❌ Tools listing failed: {e}")
        return self.test_results["tools_listing"]["status"] == "PASS"

    async def test_system_status(self, client):
        """Test system status functionality"""
        print("\n📊 Testing system status...")
        try:
            status = await client.get_system_status()
            if status and "status" in status:
                self.test_results["system_status"] = {"status": "PASS", "message": f"System status: {status.get('status', 'unknown')}"}
                print(f"   ✅ System status retrieved: {status.get('status', 'unknown')}")
                if "file_counts" in status:
                    print(f"      File counts: {status['file_counts']}")
                if "disk_usage" in status:
                    disk_pct = status['disk_usage'].get('percent_used', 0)
                    print(f"      Disk usage: {disk_pct:.1f}%")
            else:
                self.test_results["system_status"] = {"status": "FAIL", "message": "Invalid status response"}
                print("   ❌ Invalid status response")
        except Exception as e:
            self.test_results["system_status"] = {"status": "FAIL", "message": str(e)}
            print(f"   ❌ System status failed: {e}")
        return self.test_results["system_status"]["status"] == "PASS"

    async def test_document_processing(self, client):
        """Test document processing functionality"""
        print("\n📄 Testing document processing...")
        try:
            processor = ContentProcessor(client)
            test_file = self.temp_dir / "test_document.md"
            if test_file.exists():
                result = await processor.convert_document(str(test_file), "html")
                if result.get("success"):
                    self.test_results["document_processing"] = {"status": "PASS", "message": f"Document converted successfully to {result.get('target_format')}"}
                    print(f"   ✅ Document conversion successful")
                    print(f"      Output: {result.get('output_path', 'N/A')}")
                else:
                    self.test_results["document_processing"] = {"status": "FAIL", "message": result.get("error", "Unknown error")}
                    print(f"   ❌ Document conversion failed: {result.get('error', 'Unknown error')}")
            else:
                self.test_results["document_processing"] = {"status": "SKIP", "message": "Test file not found"}
                print("   ⏭️  Skipped - test file not found")
        except Exception as e:
            self.test_results["document_processing"] = {"status": "FAIL", "message": str(e)}
            print(f"   ❌ Document processing test failed: {e}")
        return self.test_results["document_processing"]["status"] == "PASS"

    async def test_image_processing(self, client):
        """Test image processing functionality"""
        print("\n🖼️  Testing image processing...")
        try:
            processor = ContentProcessor(client)
            test_image = self.temp_dir / "test_image.png"
            if test_image.exists():
                result = await processor.process_image(str(test_image), ["resize", "enhance"], width=100, height=100)
                if result.get("success"):
                    self.test_results["image_processing"] = {"status": "PASS", "message": f"Image processed with {len(result.get('operations_applied', []))} operations"}
                    print(f"   ✅ Image processing successful")
                    print(f"      Operations: {result.get('operations_applied', [])}")
                    print(f"      Output: {result.get('output_path', 'N/A')}")
                else:
                    self.test_results["image_processing"] = {"status": "FAIL", "message": result.get("error", "Unknown error")}
                    print(f"   ❌ Image processing failed: {result.get('error', 'Unknown error')}")
            else:
                self.test_results["image_processing"] = {"status": "SKIP", "message": "Test image not available"}
                print("   ⏭️  Skipped - test image not available")
        except Exception as e:
            self.test_results["image_processing"] = {"status": "FAIL", "message": str(e)}
            print(f"   ❌ Image processing test failed: {e}")
        return self.test_results["image_processing"]["status"] == "PASS"

    async def test_news_functionality(self, client):
        """Test news processing functionality"""
        print("\n📰 Testing news functionality...")
        try:
            processor = ContentProcessor(client)
            test_rss = "https://rss.cnn.com/rss/edition.rss"
            add_result = await processor.add_news_source(test_rss, "test")
            if add_result.get("success"):
                print(f"   ✅ News source added successfully")
                trending_result = await processor.get_trending_topics("24h")
                if trending_result.get("success"):
                    self.test_results["news_functionality"] = {"status": "PASS", "message": f"News functionality working, {trending_result.get('total_articles', 0)} articles found"}
                    print(f"   ✅ Trending topics retrieved ({trending_result.get('total_articles', 0)} articles)")
                else:
                    self.test_results["news_functionality"] = {"status": "PARTIAL", "message": "News source added but trending failed"}
                    print(f"   ⚠️  News source added but trending failed: {trending_result.get('error', 'Unknown error')}")
            else:
                self.test_results["news_functionality"] = {"status": "FAIL", "message": add_result.get("error", "Unknown error")}
                print(f"   ❌ News source addition failed: {add_result.get('error', 'Unknown error')}")
        except Exception as e:
            self.test_results["news_functionality"] = {"status": "FAIL", "message": str(e)}
            print(f"   ❌ News functionality test failed: {e}")
        return self.test_results["news_functionality"]["status"] == "PASS"

    async def test_resource_management(self, client):
        """Test resource management functionality"""
        print("\n📁 Testing resource management...")
        try:
            resources = await client.list_resources()
            self.test_results["resource_management"] = {"status": "PASS", "message": f"Resource listing successful, {len(resources)} resources found"}
            print(f"   ✅ Resource listing successful ({len(resources)} resources)")
            if resources:
                try:
                    content = await client.read_resource(resources[0].uri)
                    print(f"   ✅ Resource reading successful ({len(content)} characters)")
                except Exception as e:
                    print(f"   ⚠️  Resource reading failed: {e}")
        except Exception as e:
            self.test_results["resource_management"] = {"status": "FAIL", "message": str(e)}
            print(f"   ❌ Resource management test failed: {e}")
        return self.test_results["resource_management"]["status"] == "PASS"
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASS")
        failed_tests = sum(1 for result in self.test_results.values() if result["status"] == "FAIL")
        partial_tests = sum(1 for result in self.test_results.values() if result["status"] == "PARTIAL")
        skipped_tests = sum(1 for result in self.test_results.values() if result["status"] == "SKIP")
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️  Partial: {partial_tests}")
        print(f"⏭️  Skipped: {skipped_tests}")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status_emoji = {"PASS": "✅", "FAIL": "❌", "PARTIAL": "⚠️", "SKIP": "⏭️"}.get(result["status"], "❓")
            print(f"  {status_emoji} {test_name}: {result['message']}")
        
        if failed_tests == 0 and partial_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED! Server is working correctly.")
        elif failed_tests == 0:
            print(f"\n⚠️  TESTS MOSTLY PASSED with {partial_tests} partial results.")
        else:
            print(f"\n❌ {failed_tests} TESTS FAILED. Check server configuration.")
        
        print("=" * 60)
    
    async def cleanup_test_environment(self):
        """Cleanup test environment"""
        print("\n🧹 Cleaning up test environment...")
        try:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")